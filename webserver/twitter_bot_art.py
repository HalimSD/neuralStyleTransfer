import gspread
from twitter import *
import tweepy
import json


def get_keys(path):
    with open(path) as f:
        return json.load(f)


gc = gspread.service_account("credentials.json")
keys = get_keys("twitter_credentials.json")
token = keys["token"]
token_secret = keys["token_secret"]
consumer_key = keys["consumer_key"]
consumer_secret = keys["consumer_secret"]


# Open a sheet from a spreadsheet in one go
wks = gc.open("artistai").sheet1

# # Update a range of cells using the top left corner address
next_tweet = wks.acell("A2").value

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(token, token_secret)
api = tweepy.API(auth)
# user = api.get_user(screen_name="twitter")
# print(user)

# api.update_status(next_tweet)
# wks.delete_rows(2)


sender_id = ""
sender_name = ""
messages = api.get_direct_messages(count=1)
for message in reversed(messages):
    # who is sending?
    sender_id = message.message_create["sender_id"]
    sender_name = api.get_user(user_id=sender_id).name
    text = message.message_create["message_data"]["text"]

    media = api.media_upload(filename="./images/picasso.jpg")
    api.send_direct_message(
        recipient_id=sender_id,
        text="Thank You {}! Let's Get Creative. Provide Me With An Artistic Image! Like This One Above For Example!".format(sender_name),
        attachment_type="media",
        attachment_media_id=media.media_id,
    )

