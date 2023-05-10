import json
import boto3
import datetime
import time
import os
from pmaw import PushshiftAPI
from abc import ABC, abstractclassmethod, abstractmethod 
from pathlib import Path
import csv

BUCKET = os.environ['BUCKET']

SUB = os.environ['SUB']

NUM_DAYS_DELTA = int(os.environ['NUM_DAYS_DELTA'])

OUTPUT_DIR = os.environ['OUTPUT_DIR']

SCHEMA_PATH = os.environ['SCHEMA_PATH']

s3 = boto3.client('s3')


class RedditExtractor(ABC):

    def __init__(self, api : PushshiftAPI = None, num_days_delta : int = NUM_DAYS_DELTA, sub : str = SUB, schema : dict() = dict):

      self.num_days_delta = num_days_delta
      self.sub = sub
      self.schema = schema
      
      # today
      date_until = datetime.datetime.now()
      print(f"Now is {date_until}")
      date_until_start = date_until.replace(hour=0, minute=0, second=0, microsecond=0)

      # subtract one day from current day
      date_until = date_until_start - datetime.timedelta(days=self.num_days_delta - 1, seconds=1)
      date_from = date_until_start - datetime.timedelta(days=self.num_days_delta)

      print(f"dates set from {date_from} until {date_until}")

      date_from_timestamp = int(time.mktime(date_from.timetuple()))
      date_until_timestamp = int(time.mktime(date_until.timetuple()))

      print(f"timestamps set from {date_from_timestamp} until {date_until_timestamp}")

      self.date_from = date_from
      self.date_until = date_until
      self.date_from_timestamp = date_from_timestamp
      self.date_until_timestamp = date_until_timestamp

      self.api = api


    @property
    @abstractmethod
    def retrieve_api(self):
        """ Property for the method to use for reddit client based on type of data."""
        pass

    @property
    @abstractmethod
    def post_type(self):
        """ Post type of data being extracted. """
        pass
    
    @abstractmethod
    def parse_date(self, time_element):
      """ How to deal with parsing the date field based on post_type. """
      pass

    def get_posts(self):
      """Concrete method of extracting data with reddit API. """ 

      print(self.sub, self.date_from_timestamp, self.date_until_timestamp)
      t1_start = time.perf_counter()
      response = self.retrieve_api(subreddit=self.sub, since=self.date_from_timestamp, until=self.date_until_timestamp)
      t1_stop = time.perf_counter()
      
      response_size = len(response)
      
      # write the data into the '/tmp' folder.
      with open(f'/tmp/{self.post_type}.csv', 'w', encoding='utf-8', newline='\n') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(self.schema.keys())
        
        for comment in response:
          output_obj = []
          for field in self.schema.keys():
            if field == "created_utc":
              created_utc = self.parse_date(comment['created_utc'])
              output_obj.append(str(created_utc).encode("utf-8", errors='replace').decode())
            elif (field not in comment) or (comment[field] != comment[field]):
              output_obj.append(str(self.schema[field]).encode("utf-8", errors='replace').decode())
            else: 
              output_obj.append(str(comment[field]).encode("utf-8", errors='replace').decode())
          writer.writerow(output_obj)

      print(f"Wrote {response_size} lines in {t1_stop - t1_start}")
    
class CommentsExtractor(RedditExtractor):
    post_type = 'comments'

    def __init__(self, **kwds):
      super().__init__(**kwds)
    
    @property
    def retrieve_api(self):
        return self.api.search_comments
    
    def parse_date(self, time_element):
      return int(time_element)
    
class SubmissionsExtractor(RedditExtractor):
    post_type = 'submissions'

    def __init__(self, **kwds):
      super().__init__(**kwds)
    
    @property
    def retrieve_api(self):
        return self.api.search_submissions
    
    def parse_date(self, time_element):
      return str(time_element)


def lambda_handler(event, context): 

    api = PushshiftAPI()
    
    # Comments
    schema = s3.get_object(
        Bucket=BUCKET,
        Key=str(Path(SCHEMA_PATH, f"{CommentsExtractor.post_type}_schema.json")),
    )["Body"].read()
    
    comments_schema = json.loads(schema)
    # comments_schema = default_values
    
    comments_extractor = CommentsExtractor(
    api=api,
    schema=comments_schema,
    num_days_delta = NUM_DAYS_DELTA
    )
    
    # Submissions
    schema = s3.get_object(
        Bucket=BUCKET,
        Key=str(Path(SCHEMA_PATH, f"{SubmissionsExtractor.post_type}_schema.json")),
    )["Body"].read()
    
    submissions_schema = json.loads(schema)
    # comments_schema = default_values
    
    submissions_extractor = SubmissionsExtractor(
    api=api,
    schema=submissions_schema,
    num_days_delta = NUM_DAYS_DELTA
    )
    

    comments_extractor.get_posts()
    
    comments_directory_path = comments_extractor.date_from.strftime("year=%Y/month=%m/day=%d")
    
    comments_key = str(
    Path(
        OUTPUT_DIR,
        CommentsExtractor.post_type,
        comments_directory_path, 
        f"{CommentsExtractor.post_type}.csv"
    )
    )
    
    s3.upload_file(
    f"/tmp/{CommentsExtractor.post_type}.csv",
    BUCKET,
    comments_key
    )
    
    print(f"Saved to {comments_key}")

    submissions_extractor.get_posts()
    
    submissions_directory_path = submissions_extractor.date_from.strftime("year=%Y/month=%m/day=%d")
    
    submissions_key = str(
    Path(
        OUTPUT_DIR,
        SubmissionsExtractor.post_type,
        submissions_directory_path, 
        f"{SubmissionsExtractor.post_type}.csv"
    )
    )
    
    s3.upload_file(
    f"/tmp/{SubmissionsExtractor.post_type}.csv",
    BUCKET,
    submissions_key
    )
    
    print(f"Saved to {submissions_key}")
    
