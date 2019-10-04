#!/usr/bin/env python3
# coding=utf-8
import datetime
import os
import sys
import dataset
import ccxt
from time import sleep
import requests
import json
import ssl
import xmltodict
# import urllib2
import time
print(os.__file__)
print(ccxt.__file__)

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

mode = 'red'

msec = 1000
minute = 60 * msec
hold = 30
exchange_dict = {
    'binance': [
'BTC/USDT',
'ETH/USDT',
'TRX/BTC',
'ADA/BTC',
'XLM/BTC',
'XRP/BTC',
'IOTA/BTC',
'KMD/BTC',
'NANO/BTC',
'EOS/BTC',
'BNB/BTC',
'NEO/BTC',
'LTC/BTC',
'XMR/BTC',
'DASH/BTC',
'ETH/BTC',
'XEM/BTC',
'QTUM/BTC',
'DCR/BTC',
'ZEC/BTC'
]
}

exchange_list = ['binance']

ticker_exchange_list = []

binance_currency = [
'BTC/USDT',
'ETH/USDT',
'TRX/BTC',
'ADA/BTC',
'XLM/BTC',
'XRP/BTC',
'IOTA/BTC',
'KMD/BTC',
'NANO/BTC',
'EOS/BTC',
'BNB/BTC',
'NEO/BTC',
'LTC/BTC',
'XMR/BTC',
'DASH/BTC',
'ETH/BTC',
'XEM/BTC',
'QTUM/BTC',
'DCR/BTC',
'ZEC/BTC'
]

bitmex_currency = ['']



# DEFINE DATABASE CONNECTIONS HERE FOR EACH EXCHANGE
# Using different databases for each 


binance_db = dataset.connect('mysql://bender_user:Paswurd2@206.189.193.88/binance')


binance = ccxt.binance({
    'rateLimit': 3000,
    'enableRateLimit': True,
    # 'verbose': True,
    'exchangeName': "binance"
    , 'database': binance_db
})


def koineks_ticker(currency):
    # To use this function, instead of calling a currency pair, just use BTC, ETH, XLM, LTC, DASH,XRP,DOGE

    # Because of pair api response is different than other api calls, there is another dict called koineks_pair to normalize it.

    try:

        data = requests.get('https://koineks.com/ticker')
        high = data.json()[currency]['high']
        low = data.json()[currency]['low']
        closing = float(data.json()[currency]['current'])
        volume = data.json()[currency]['volume']
        timestamp = data.json()[currency]['timestamp']
        ask = data.json()[currency]['ask']
        bid = data.json()[currency]['bid']

        # Check if there is empty values. Sometimes, some apis doesnt have values
        # and this causes problems in database records.

        if not high:
            high = 0
        if not low:
            low = 0
        if not volume:
            volume = 0
        if not bid:
            bid = 0
        if not ask:
            ask = 0

        delta_value = delta_koineks(currency)

        price_delta_1h = delta_value[0]

        price_delta_24h = delta_value[1]

        print(str(currency) + " Koineks Delta 1h: " + str(price_delta_1h))
        print(str(currency) + " Koineks Delta 24h: " + str(price_delta_24h))

        trading_pairs = {
            "trading_pair_id": 'koineks' + '_' + str(koineks_pair[currency]).replace("/", "_").lower(),
            "trading_pair": koineks_pair[currency],
            "price": closing,
            "price_delta_1h": price_delta_1h,
            "price_delta_24h": price_delta_24h,
            "price_updated_at": timestamp
        }
        sleep(0.035)

        database_write('koineks', currency, timestamp, 0, high, closing, low, volume, ask, bid)

    except Exception as error:
        print('got an error in koineks ticker fetching' + str(error))
    return (trading_pairs)


def koineks_update(pairs):
    firebase_pairs = []

    firebase_payload = {
        "id": 'koineks',
        "name": 'KOINEKS',
        "trading_pairs": firebase_pairs
    }
    # Cycle through pairs to push prices to firebase.
    try:
        for x in pairs:
            data = koineks_ticker(x)
            print("Fetching data from Koineks for " + str(x))
            firebase_pairs.append(data)
        firebase_prices_push(firebase_payload)

    except Exception as error:
        print("error in koineks ticker fetching. Error: " + str(error))


def candle24h(exchange, currency):
    try:
        start_date = datetime.datetime.utcnow() + datetime.timedelta(-1)
        # following line is in milliseconds
        # kraken has a different candle response. Because of this there should be an additional code to check
        # what is the timestamp order is.
        from_timestamp = exchange.parse8601(str(start_date))
        candles = exchange.fetch_ohlcv(currency, '5m', from_timestamp, 288)

        # check the candle order by looking at the timestamps of the list.
        if candles[0][0] < candles[-1][0]:
            opening = candles[-1][1]
            high = candles[-1][2]
            low = candles[-1][3]
            closing = candles[-1][4]
            volume = candles[-1][5]

            price_1h = candles[-11][4]
            price_24h = candles[0][4]
            delta1h = (closing - price_1h) / closing * 100
            delta24h = (closing - price_24h) / closing * 100
            timestamp = candles[-1][0]


        else:
            opening = candles[0][1]
            high = candles[0][2]
            low = candles[0][3]
            closing = candles[0][4]
            volume = candles[0][5]

            price_1h = candles[11][4]
            price_24h = candles[-1][4]
            delta1h = (closing - price_1h) / closing * 100
            delta24h = (closing - price_24h) / closing * 100
            timestamp = candles[0][0]

        trading_pairs = {
            "trading_pair_id": id_convert(exchange, currency),
            "trading_pair": currency,
            "price": float(closing),
            "price_delta_1h": float(delta1h),
            "price_delta_24h": float(delta24h),
            "price_updated_at": timestamp
        }

        # Write all that shit into database

        database_write(exchange, currency, timestamp, opening, high, closing, low, volume, 0, 0)

        sleep(exchange.rateLimit / 1000)

    except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:

        print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')

        sleep(exchange.rateLimit / 1000)

    return trading_pairs


def ticker(exchange, currency):
    try:
        tickers = exchange.fetch_tickers(currency)
        ticker = tickers[currency]
        print(ticker, type(ticker))
        closing = ticker['last']
        timestamp = ticker['timestamp']
        open = ticker['open']
        high = ticker['high']
        low = ticker['low']
        volume = ticker['baseVolume']
        bid = ticker['bid']
        ask = ticker['ask']

        # Check if there is empty values. Sometimes, some apis doesnt have values
        # and this causes problems in database records.

        if not open:
            open = 0
        if not high:
            high = 0
        if not low:
            low = 0
        if not volume:
            volume = 0
        if not bid:
            bid = 0
        if not ask:
            ask = 0

        trading_pairs = {
            "trading_pair_id": id_convert(exchange, currency),
            "trading_pair": currency,
            "price": float(closing),
            # "price_delta_1h": price_delta_1h,
            # "price_delta_24h": price_delta_24h,
            "price_updated_at": timestamp
        }
        print("writing", exchange, currency, timestamp, open, high, closing, low, volume, bid, ask)
        database_write(exchange, currency, timestamp, open, high, closing, low, volume, bid, ask)

        sleep(exchange.rateLimit / 1000)
    except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
        print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')
        sleep(exchange.rateLimit / 1000)
    return trading_pairs


def candle_update(exchange, pairs):
    firebase_pairs = []

    firebase_payload = {
        "id": str(exchange.exchangeName),
        "name": str(exchange.exchangeName).upper(),
        "trading_pairs": firebase_pairs
    }

    try:
        for x in pairs:
            data = candle24h(exchange, x)
            print("Fetching data from " + str(exchange) + " for " + str(x))
            firebase_pairs.append(data)
        firebase_prices_push(firebase_payload)

    except Exception as error:
        print("error " + str(exchange) + " in candle fetching. Error: " + str(error))


def ticker_update(exchange, pairs):

    try:
        for x in pairs:
            #Secondary try exception to catch the exception and continue looping.
            try:
                data = ticker(exchange, x)
                print("writing data for " + str(x))
                # firebase_pairs.append(data)
            except Exception as error:
                print(str(error))
                sleep(exchange.rateLimit / 1000)
                continue

        # firebase_prices_push(firebase_payload)

    except Exception as error:
        print("error " + str(exchange) + " in ticker fetching. Error: " + str(error))

# This function is optional, if you would like to use firebase datastore
def firebase_exchanges():
    payload_2 = []

    for x in xrange(13):
        payload_1 = {
            "id": exchange_list[x],
            "name": exchange_list[x],
            "trading_pairs": exchange_dict[exchange_list[x]],
        }
        payload_2.append(payload_1)
    payload_out = json.dumps(payload_2)
    headers = {
        'content-type': "application/json"
    }

    url = "your_url_for_firebase"
    response = requests.request("POST", url, data=payload_out, headers=headers)
    print(response.text)
    print(payload_2)
    print("payload generated for: " + str(x))


def id_convert(exchange, pair):
    id = str(exchange.exchangeName) + '_' + str(pair).replace("/", "_").lower()
    return id

# This function is optional, if you would like to use firebase datastore
def firebase_prices_push(data):
    try:
        payload_out = json.dumps(data)
        headers = {
            'content-type': "application/json"
        }

        url = "firebase_url"
        response = requests.request("POST", url, data=payload_out, headers=headers)
        print(response.text)
        print("payload generated")
    except Exception as error:
        print ("Error occured while pushing to Firebase " + str(error))

# Fetch Euro prices for arbitrage opportunities from different exchanges from different countries.

def ecb_fetch():
    try:
        # Fetch XML data from European Central Exchange for EUR Fiat currencies.

        file = urllib2.urlopen('http://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml')
        data = file.read()
        file.close()
        result = xmltodict.parse(data)

        # Filter the data of the dictionary get rid of un useful parts.

        filtered_data = result['gesmes:Envelope']['Cube']['Cube']

        timestamp_date = filtered_data['@time']

        firebase_pairs = []

        firebase_payload = {
            "id": str("ecb"),
            "name": str("ECB"),
            "trading_pairs": firebase_pairs
        }
        for x in xrange(32):
            currency = filtered_data['Cube'][x]['@currency']
            closing = float(filtered_data['Cube'][x]['@rate'])
            trading_pairs = {
                "trading_pair_id": 'ecb' + '_' + 'eur_' + str(currency).lower(),
                "trading_pair": 'EUR/' + str(currency),
                "price": closing,
                "price_delta_1h": None,
                "price_delta_24h": None,
                "price_updated_at": int(round(time.time() * 1000))  # ms timestamp
            }
            firebase_pairs.append(trading_pairs)
        firebase_prices_push(firebase_payload)
        print("Fetched and pushed European Central Bank Pairs into Firebase")
    except Exception as error:
        print ("error in European Central Bank Data Fetching" + str(error))


def database_write(exchange, currency, timestamp, opening, high, close, low, volume, bid, ask):
    try:
        
        exchange.database.begin()

        exchange.database[currency + mode].insert(dict(time_ms=int(timestamp/1000), opening=float(opening), high=float(high),
                                                close=float(close), low=float(low), volume=float(volume),
                                                bid=float(bid),
                                                ask=float(ask)))

        try:
            exchange.database.commit()
        except Exception:
            session.rollback()
    except Exception as error:
        print("error in database_write function " + str(error))


def ticker_delta(exchange, currency):
    try:
        table = exchange.database[currency]

        ticker_length = len(table)

        # Check if there is enough candles in database for delta calculation.

        if ticker_length > 10:

            time_between_tickers_min = (float(table.find_one(id='10')['time_ms']) - float(
                table.find_one(id='1')['time_ms'])) / 10 / 60000

            delta1h_ticker_count = int(time_between_tickers_min * 60)

            delta24h_ticker_count = int(time_between_tickers_min * 60 * 24)

            last_ticker = table.find_one(id=str(ticker_length))['close']

        else:

            delta1h_ticker_count = 0

            delta24h_ticker_count = 0

            last_ticker = 1

        print ("delta1h ticker count " + str(delta1h_ticker_count))
        print ("delta24h ticker count " + str(delta24h_ticker_count))

        if 0 < delta24h_ticker_count < ticker_length:

            delta24h_ticker = int(table.find_one(id=str(ticker_length - delta24h_ticker_count))['close'])

            delta24h = ((last_ticker - delta24h_ticker) / last_ticker) * 100

        else:

            delta24h = None

        if 0 < delta1h_ticker_count < ticker_length:

            delta1h_ticker = int(table.find_one(id=str(ticker_length - delta1h_ticker_count))['close'])

            delta1h = ((last_ticker - delta1h_ticker) / last_ticker) * 100

        else:

            delta1h = None
        print (str(currency) + " delta 1h: " + str(delta1h))
        print (str(currency) + " delta 24h: " + str(delta24h))

    except Exception as error:
        print (error)
        delta24h = None
        delta1h = None
    return (delta1h, delta24h)


def delta(exchange, currency):
    try:
        exchange.database.begin()

        table = exchange.database[currency]

        table_length = len(table)

        global delta_1h
        delta_1h = 0

        global delta_24h
        delta_24h = 0


        date_1h = datetime.datetime.utcnow()

        date_24h = datetime.datetime.utcnow() + datetime.timedelta(-1)

        timestamp_1h = exchange.parse8601(str(date_1h)) - 3600000

        timestamp_24h = exchange.parse8601(str(date_24h))

        result_1h = table.find_one(table.table.columns.time_ms > timestamp_1h)

        result_24h = table.find_one(table.table.columns.time_ms > timestamp_24h)

        result_now = table.find_one(id=table_length)

        closing_now = result_now['close']
        print("test")
        # CHECK IF THE VALUES ARE NONE TYPE OR NOT.
        if not result_1h:
            print("none value for result_1h")
            delta_1h = 0
        else:
            closing_1h = result_1h['close']
            delta_1h = ((closing_now - closing_1h) / closing_now) * 100
        
        if not result_24h:
            print("none value for result_24h")
            delta_24h = 0
        else:
            closing_24h = result_24h['close']
            delta_24h = ((closing_now - closing_24h) / closing_now) * 100

        try:
            exchange.database.commit()
        except Exception:
            session.rollback()
    except Exception as error:
        print("Error in Delta function: " + str(error))
    return (delta_1h, delta_24h)


def delta_koineks(currency):
    try:
        koineks_db.begin()
        table = koineks_db[currency]

        table_length = len(table)

        # Koineks has a strange timestamp results thats why I divide it to 1000.

        timestamp_1h = int((round(time.time() * 1000) - 3600000) / 1000)

        timestamp_24h = int((round(time.time() * 1000) - 86400000) / 1000)

        result_1h = table.find_one(table.table.columns.time_ms > timestamp_1h)

        result_24h = table.find_one(table.table.columns.time_ms > timestamp_24h)

        result_now = table.find_one(id=table_length)

        closing_now = result_now['close']

        # CHECK IF THE VALUES ARE NONE TYPE OR NOT.
        if not result_1h:
            print("none value for result_1h")
            delta_1h = 0
        else:
            closing_1h = result_1h['close']
            delta_1h = ((closing_now - closing_1h) / closing_now) * 100

        if not result_24h:
            print("none value for result_24h")
            delta_24h = 0
        else:
            closing_24h = result_24h['close']
            delta_24h = ((closing_now - closing_24h) / closing_now) * 100
        koineks_db.commit()
    except Exception as error:
        print("Error in  delta_koineks function: " + str(error))
    return (delta_1h, delta_24h)


def update_all():
    while (True):
        try:
            ticker_update(binance, binance_currency)
        except:
            print("error in update all function or user quit")
            sys.exit(0)

update_all()

