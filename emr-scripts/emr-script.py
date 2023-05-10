from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, ArrayType, IntegerType, StructType, StructField, DoubleType, DateType

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, NGram
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.ml.stat import Summarizer
from pyspark.ml import Pipeline

import datetime
import numpy as np
import string, re
import argparse
from pathlib import Path
import os

from statsmodels.tsa.seasonal import seasonal_decompose

class CustomTokenizer(Tokenizer):
    def _transform(self, dataset):
        non_space_whitespace = string.whitespace.replace(" ", "")

        @udf(returnType=ArrayType(StringType()))
        def remove_punctuation_and_tokenize(text):
            if text is None:
              return []
            # Remove punctuation
            cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
            # cleaned_text = cleaned_text.translate(str.maketrans('', '', string.digits))
            cleaned_text = re.sub(f"[{non_space_whitespace}]", " ", cleaned_text)
            
            # Tokenize (split by whitespace)
            tokens = cleaned_text.split()
            
            return tokens
        
        return dataset.withColumn(self.getOutputCol(), remove_punctuation_and_tokenize(dataset[self.getInputCol()]))

@udf(returnType=VectorUDT())
def trim_sparse_vector(v, split_index):
  where = np.where(v.indices > split_index)
  if where[0].size > 0:
    cutoff = where[0][0]
    v.indices = v.indices[:cutoff]
    v.values = v.values[:cutoff]
  v.size = split_index + 1
  return v

@udf(returnType=DateType())
def convert_week_num(week_num, ref_date):
  return (datetime.datetime.strptime(ref_date, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=(week_num - 1)*7))

def find_in_sparse_vector(raw_feature, terms_indices, term_values, term_peak_times):
  return [(term_id, int(raw_feature[term_id]), value, peak_time) for (term_id, value, peak_time) in zip(terms_indices, term_values, term_peak_times) if term_id in raw_feature.indices]

def decompose_rdd(timeseries_sv, period, resid_std_threshold, min_counts, model_type):
  id = timeseries_sv[0]
  ts = timeseries_sv[1].toArray()[1:]
  sd = seasonal_decompose(ts, two_sided=False, model=model_type, period=period)
  trend, seasonal, resid = sd.trend[~np.isnan(sd.trend)], sd.seasonal[~np.isnan(sd.seasonal)], sd.resid[~np.isnan(sd.resid)]

  sum_variances = np.var(trend) + np.var(seasonal) + np.var(resid)
  ratio_season = np.var(seasonal) / sum_variances
  ratio_trend = np.var(trend) / sum_variances
  ratio_resid = np.var(resid) / sum_variances

  mask = np.multiply((resid > (np.mean(resid) + resid_std_threshold * np.std(resid))), (resid > min_counts))

  return [id, np.where(mask)[0] + period + 1, ratio_season, ratio_trend, ratio_resid]

def parse_args():
    parser = argparse.ArgumentParser()
      
    parser.add_argument('-s3_source_path')
    parser.add_argument('-s3_result_path')
    parser.add_argument('-s3_result_table_path')
    parser.add_argument('-stop_words_path')
    parser.add_argument('-start_year', type=int, default=2013)
    parser.add_argument('-min_count_threshold_total', type=float, default=200.)
    parser.add_argument('-num_weeks', type=int, default=574)
    parser.add_argument('-num_partitions', type=int, default=12)
    parser.add_argument('-sample_size', type=float, default=0.1)
    parser.add_argument('-std_band_size', type=int, default=5)
    parser.add_argument('-period', type=int, default=52)
    parser.add_argument('-min_count_threshold_week', type=int, default=50)
    parser.add_argument('-decompose_model_type', default="additive")

    args = parser.parse_args()
    return args

def main():
    conf = SparkConf().setAppName("AWS-project")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    args = parse_args()

    REF_DATE = str(datetime.datetime(args.start_year, 1, 1, 0, 0, 0))
    CURRENT_TIMESTAMP = datetime.datetime.today().strftime("%m-%d-%Y-%H:%M:%S")

    comments_df = spark.read.option("header","true") \
                        .option("recursiveFileLookup","true") \
                        .parquet(f"{args.s3_source_path}/comments_6/") \
                        .sample(args.sample_size)

    submissions_df = spark.read.option("header","true") \
                            .option("recursiveFileLookup","true") \
                            .parquet(f"{args.s3_source_path}/submissions_3/") \
                            .sample(args.sample_size)
    

    # comments_df = (spark.read
    #     .option("multiline", "true")
    #     .option("quote", '"')
    #     .option("header", "true")
    #     .option("escape", "\\")
    #     .option("escape", '"')
    #     .csv("/content/drive/MyDrive/proj/uwaterloo_comments (1).csv")
    # ).sample(0.01, seed=441)

    # submissions_df = (spark.read
    #     .option("multiline", "true")
    #     .option("quote", '"')
    #     .option("header", "true")
    #     .option("escape", "\\")
    #     .option("escape", '"')
    #     .csv("/content/drive/MyDrive/proj/uwaterloo_submissions (1).csv")
    # ).sample(0.1, seed=441)

    print(f"num_comments, num_submissions: {comments_df.count()}, {submissions_df.count()}")

    submissions_subset = (submissions_df.withColumn("created_utc", from_unixtime("created_utc"))
        .filter((year(col("created_utc")) >= args.start_year))
        .withColumn("text", concat_ws(' ', "selftext", "title"))
        .withColumn('week', ceil(datediff('created_utc', lit(REF_DATE))/7))
        .select("created_utc", "author", "text", "score", col('id').alias("link_id"), "week")
    )

    submissions_title = (submissions_df.withColumn("created_utc", from_unixtime("created_utc"))
        .filter((year(col("created_utc")) >= args.start_year))
        .withColumn('week', ceil(datediff('created_utc', lit(REF_DATE))/7))
        .select("created_utc", "author", "title", "score", col('id').alias("title_link_id"), "week")
    )

    comments_subset = (comments_df.withColumn("created_utc", from_unixtime("created_utc"))
        .filter((year(col("created_utc")) >= args.start_year))
        .withColumn("link_id", expr("substring(link_id, 4, length(link_id))"))
        .withColumn('week', ceil(datediff('created_utc', lit(REF_DATE))/7))
        .select("created_utc", "author", col('body').alias("text"), "score", "link_id", "week")
    )

    # submissions_subset = (submissions_df
    #     .filter((year(col("created_utc")) >= 2013) & (year(col("created_utc")) < 2023))
    #     .withColumn("text", concat_ws(' ', "selftext", "title"))
    #     .withColumn('week', ceil(datediff('created_utc', lit(REF_DATE))/7))
    #     .select("created_utc", "author", "text", "score", col('id').alias("link_id"), "week")
    # )

    # submissions_title = (submissions_df
    #     .filter((year(col("created_utc")) >= 2013) & (year(col("created_utc")) < 2023))
    #     .withColumn('week', ceil(datediff('created_utc', lit(REF_DATE))/7))
    #     .select("created_utc", "author", "title", "score", col('id').alias("title_link_id"), "week")
    # )

    print(f"subset_comments, subset_submissions: {comments_subset.count()}, {submissions_subset.count()}")

    df_concat = comments_subset.union(submissions_subset)
    df_concat.cache()

    print(f"total data size: ", df_concat.count())

    stop_words = spark.sparkContext.textFile(os.path.join(args.stop_words_path, 'stop_words')).collect()

    tokenizer = CustomTokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)

    pipeline = Pipeline(stages=[tokenizer, remover])

    # Fit the pipeline to the data
    model = pipeline.fit(df_concat)

    # Transform the data
    processed_data = model.transform(df_concat)

    aggregated_processed_data = (processed_data
        .groupby('week')
        .agg(
            flatten(collect_list('filtered_tokens')).alias("filtered_tokens"),
            flatten(collect_list('tokens')).alias("unfiltered_tokens") 
        )
    )

    ngram = NGram(n=2, inputCol="filtered_tokens", outputCol="bigrams")
    ngrams_df = ngram.transform(aggregated_processed_data)
    ngrams_df = ngrams_df.withColumn("combined_uni_bigrams", concat(col("filtered_tokens"), col("bigrams")))

    count_vectorizer = CountVectorizer(
        inputCol="combined_uni_bigrams",
        outputCol="raw_features",
        minDF=10,
        vocabSize=2 ** 20
    )

    model = count_vectorizer.fit(
        ngrams_df
    )

    print(f"Vocab size and model size: {model.getVocabSize()}, {len(model.vocabulary)}")

    vectorized_data = model.transform(ngrams_df)

    ngram_original = NGram(n=2, inputCol="tokens", outputCol="comment_bigrams")
    comment_ngrams_df = ngram_original.transform(processed_data)
    comment_ngrams_df = comment_ngrams_df.withColumn("combined_uni_bigrams", concat(col("filtered_tokens"), col("comment_bigrams")))

    individual_level_vectorized_data = model.transform(comment_ngrams_df)

    sv = vectorized_data.select(Summarizer.normL1(vectorized_data.raw_features)).take(1)
    sv_array = sv[0]['normL1(raw_features)'].toArray()
    print(f"Frequencies for normL1: {sv_array[:10]}")
    split_index = np.where(sv_array == args.min_count_threshold_total)[0][-1]

    print(f"split index : {split_index}")

    trimmed_vectors = vectorized_data.withColumn(
        "trimmed_features",
        trim_sparse_vector(col('raw_features'), lit(int(split_index)))
    )

    transformed_rdd = trimmed_vectors.rdd \
        .flatMap(lambda row: [(term_id, (row['week'], term_count)) for (term_id, term_count) in zip(row["trimmed_features"].indices, row["trimmed_features"].values)]) \
        .groupByKey() \
        .mapValues(lambda x: sorted(list(x), key=lambda y: y[0])) \
        .map(lambda x: (x[0], SparseVector(args.num_weeks + 1, [i[0] for i in x[1]], [i[1] for i in x[1]])))

    decomposed_rdd = transformed_rdd.map(
        lambda x: decompose_rdd(
            x, 
            period=args.period,
            resid_std_threshold=args.std_band_size,
            min_counts=args.min_count_threshold_week,
            model_type=args.decompose_model_type
        )
    ).filter(lambda x: x[1].size > 0)

    outlier_terms = decomposed_rdd.collect()

    ######
    decomposed_rdd_data = []

    decomposed_rdd_schema = StructType([
        StructField("term_id", IntegerType(), True),
        StructField("term_name", StringType(), True),
        StructField("peak_weeks", ArrayType(IntegerType()), True),
        StructField("ratio_season", DoubleType(), True),
        StructField("ratio_trend", DoubleType(), True),
        StructField("ratio_resid", DoubleType(), True),
    ])

    subset_vocab = model.vocabulary[:split_index + 1]

    for element in outlier_terms:
        decomposed_rdd_data.append((
            int(element[0]),
            str(subset_vocab[element[0]]),
            element[1].tolist(), 
            float(element[2]), 
            float(element[3]), 
            float(element[4])
        ))

    decomposed_rdd_df = spark.createDataFrame(
        decomposed_rdd_data,
        schema=decomposed_rdd_schema
    )

    decomposed_rdd_df.write.parquet(f"{args.s3_result_path}/{CURRENT_TIMESTAMP}/decompose_results")  
    
    selected_outlier_term_indices = list(map(lambda x: int(x[0]), outlier_terms))

    print("Matching vocabulary: ")
    selected_outlier_terms_values = [subset_vocab[i] for i in (selected_outlier_term_indices)]

    selected_outlier_terms_peak_times = list(map(lambda x: int(x[1][0]), outlier_terms))

    print(f"final selected indices size {len(selected_outlier_term_indices)}")
    print(f"Sample final values: {selected_outlier_terms_values[:10]}")

    find_in_sparse_vector_result_type = ArrayType(StructType([
        StructField("term_id", IntegerType(), False),
        StructField("count", IntegerType(), False),
        StructField("term", StringType(), False),
        StructField("peak_time", IntegerType(), False),
    ]))

    def find_in_sparse_vector_udf(terms_indices, terms_values, term_peak_times):
        return udf(lambda l: find_in_sparse_vector(l, terms_indices, terms_values, term_peak_times), find_in_sparse_vector_result_type)

    terms_contained_in_each_document = individual_level_vectorized_data \
        .withColumn(
                "term_ids_contained", 
                find_in_sparse_vector_udf(
                    selected_outlier_term_indices, 
                    selected_outlier_terms_values,
                    selected_outlier_terms_peak_times
                )(col('raw_features'))
            ) \
        .filter(size('term_ids_contained') > 0)

    # 1911 rows, with test sample sizes, 1932 exploded
    result_df = terms_contained_in_each_document \
        .select(
            'link_id',
            'week', 
            'author', 
            'created_utc', 
            'term_ids_contained', 
            'text',
            explode(col('term_ids_contained')).alias('term_ids_contained_explode')
        ) \
        .withColumn('term_id', col('term_ids_contained_explode').term_id) \
        .withColumn('count', col('term_ids_contained_explode').count) \
        .withColumn('term', col('term_ids_contained_explode').term) \
        .withColumn('peak_time', col('term_ids_contained_explode').peak_time) \
        .groupBy('link_id', 'week', 'term_id') \
        .agg(
            sum('count').alias('num_occurrences_agg'), \
            first('term').alias('term'),
            first('peak_time').alias('peak_time'),
            count('term').alias('term_count')
            ) \
        .withColumnRenamed('link_id', 'agg_link_id') \
        .withColumnRenamed('week', 'agg_week') \
        .join(submissions_title, col('agg_link_id') == submissions_title.title_link_id) \
        .select('agg_link_id', 'num_occurrences_agg', 'author', 'title', 'agg_week', 'term_id', 'term', 'term_count', 'peak_time') \
        .withColumn('converted_week', convert_week_num('agg_week', lit(REF_DATE)))


    print(f"RDD Num partitions: {result_df.rdd.getNumPartitions()}")

    # result_df = result_df.coalesce(NUM_PARTITIONS)

    # print(f"RDD Num partitions after coalesce: {result_df.rdd.getNumPartitions()}")

    result_df.write.parquet(f"{args.s3_result_path}/{CURRENT_TIMESTAMP}/final_results")  

    df = spark.createDataFrame(
        data=[
            ("emr-results", CURRENT_TIMESTAMP)
        ],
        schema=StructType([
            StructField("table_type", StringType(), True),
            StructField("run_time", StringType(), True),
        ])
    )

    df.coalesce(1).write.format('csv').mode("overwrite").save(args.s3_result_table_path)

if __name__ == '__main__':
    main()
    
