#!/usr/bin/env python3
import requests
import datetime
from binance.client import Client
import pandas as pd

binance = Client("vg1Sdkl5oB09uaSIKrOoRHyl1xTJCtem9UgzpFbjFn40YHrhBRw5Iwztu111Hk9L", "cE8fd8CSEhHXCbrqL8M3BUvxcYiIoJqQTcuCm1o18Xs4S0aAYdrRohAqgbcsxBmn")

symbols = [
    'IOTABTC',
    'BCHABCBTC',
    'XLMBTC',
    'ETHBTC',
    'ETHUSDT',
    'NEOBTC',
    'BTCUSDT',
    'EOSBTC',
    'LTCBTC',
    'ADABTC',
    'TRXBTC',
    'XRPBTC',
    'NANOBTC',

]

tuples = {}
# Open time
# Open
# High
# Low
# Close
# Volume
# Close time
# Quote asset volume
# Number of trades
# Taker buy base asset volume
# Taker buy quote asset volume
# Can be ignored

test = ['IOTABTC', 'IOTAUSDT']
for symbol in test:
    kline = binance.get_historical_klines(symbol, Client.KLINE_INTERVAL_3MINUTE, str(datetime.date.today() - datetime.timedelta(7)))
    df = pd.DataFrame

    del kline[-1]
    
    print(kline[0])
    del kline[-6]
    del kline[-2]
    print(kline[0])
    tuples[symbol] = kline

url = "https://www.streamr.com/api/v1"

# login
r = requests.post(url=url + "/login/apikey", json={"apiKey": "YBgaM2MrT-66sFmo0Bu83g-8LSQpJbREO9PMrfcVbp6A"})
response = r.json()
print(response)

headers = {"Authorization": "Bearer " + response['token']}

# streams

r = requests.get(url=url + "/streams", headers=headers)
streams = r.json()

news = [
            'Finsents Crypto-NANOTOKEN News Stream',
            'Finsents Crypto-NANOTOKEN Daily Stream',
            'Finsents Crypto-NANOTOKEN Stream',
            'Finsents Crypto-Litecoin Daily Stream',
            'Finsents Crypto-Litecoin Stream',
            'Finsents Crypto-Litecoin News Stream',
            'Finsents Crypto-Ethereum Stream',
            'Finsents Crypto-Ethereum Daily Stream',
            'Finsents Crypto-Ethereum News Stream',
            'Finsents Crypto-Bitcoin Stream',
            'Finsents Crypto-Bitcoin News Stream',
            'Finsents Crypto-Bitcoin Daily Stream',
            'Finsents Crypto-Ripple Stream',
            'Finsents Crypto-Ripple News Stream',
            'Finsents Crypto-Ripple Daily Stream',
        ]

commits = [
                'GitHub commits EOSIO/eos',
                'GitHub commits iotaledger/iri',
                'GitHub commits litecoin-project/litecoin',
                'GitHub commits input-output-hk/cardano-sl',
                'GitHub commits ethereum/go-ethereum',
                'GitHub commits stellar/stellar-core',
                'GitHub commits bitcoin/bitcoin',
                'GitHub commits tronprotocol/java-tron',
                'GitHub commits ripple/rippled'
            ]

tweets = [
            'IOTA Tweets',
            'Bitcoin Cash Tweets',
            'Stellar Tweets',
            'Ethereum Tweets',
            'NEO Tweets',
            'Bitcoin Tweets',
            'EOS Tweets',
            'Litecoin Tweets',
            'Cardano Tweets',
            'Tron Tweets',
            'Ripple Tweets',
        ]

analyses = [
            '15Min',
            '1H',
            'Chart',
            '1Day',
            '4H',
        ]

telegram = ['Telegram chat member counts']
            
notifications = ['Notifications']
            


for stream in streams:
    if stream['name'] in tweets:
        print("Tweet source: ", stream['name'], "\n\n")

        last_week = datetime.date.today() - datetime.timedelta(7)
        week_unix = last_week.strftime("%s")
        today_unix = datetime.date.today().strftime("%s")

        r = requests.get(url=url + "/streams/" + stream['id'] + "/data/partitions/0/last?fromTimestamp=" + week_unix + "?toTimestamp=" + today_unix,
                            headers=headers)
        data = r.json()
        # if(len(data) != 0):
            # analyze_tweet(data)

# def analyze_tweet(data):

