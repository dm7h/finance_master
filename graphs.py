import pathlib

import requests
import urllib
import json

import sklearn
from IPython.core.display import display, HTML, JSON
from types import SimpleNamespace
import logging
import plotly.graph_objects as go
import pandas as pd
import pandas_ta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET

display(HTML("<style>.container { width:90% !important; }</style>"))


class FinancialDataAPI:
    def __init__(self):
        self.url = 'https://web.api.six-group.com/api/findata'

        self.headers = {
            "content-type": "application/json",
            "accept": "application/json",
            "api-version": "2022-06-01"
        }
        self.session = requests.session()
        certificate_path = '/home/dmin7/Downloads/API certificates/ch52991-hackathon9'
        self.session.cert = (f'{certificate_path}/signed-certificate.pem', f'{certificate_path}/private-key.pem')

    def http_request(self, end_point: str, query_string: dict) -> str:
        # Make an HTTP request and send the raw response
        try:
            http_request = f"{self.url}{end_point}?{urllib.parse.urlencode(query_string)}"

            r = self.session.get(http_request, headers=self.headers)  # , verify='./six-certificate/certificate.pem')
            if str(r.status_code)[0] != "2":
                logging.debug(f"HTTP{r.status_code}: {r.content}")
            else:
                logging.debug(f"HTTP{r.status_code}: {json.dumps(json.loads(r.content), indent=2)}")

            return r
        except requests.exceptions.SSLError as err:
            logging.error(f"Error - {http_request}:\r\n{err}")
            raise (Exception(str(err)))

    def http_request_with_scheme_id(self, end_point: str, scheme: str, ids: list) -> str:
        query_string = query_string = {
            'scheme': scheme,
            'ids': ",".join(ids)
        }
        return self.http_request(end_point, query_string)

    def _convert_response_to_object(self, http_response):
        if str(http_response.status_code)[0] == "2":
            obj = json.loads(http_response.content, object_hook=lambda d: SimpleNamespace(**d))
            return obj
        return None

    def text_search(self, query: str) -> object:
        end_point = "/v1/searchInstruments"
        # end_point = "/search/v1/"
        query_string = {'query': query}
        resp = self.http_request(end_point, query_string)

        return self._convert_response_to_object(resp)

    def instrument_summary(self, scheme: str, instruments: list):
        end_point = "/v1/instruments/referenceData/instrumentSummary"
        # end_point = "/v1/summary/instruments"
        resp = self.http_request_with_scheme_id(end_point, scheme, instruments)
        return self._convert_response_to_object(resp)

    def instrument_symbology(self, scheme: str, instruments: list):
        end_point = "/v1/instruments/referenceData/instrumentSymbology"
        resp = self.http_request_with_scheme_id(end_point, scheme, instruments)
        return self._convert_response_to_object(resp)

    def instrument_BASELIII_HQLA_EU(self, scheme: str, instruments: list):
        end_point = "/v1/instruments/_regulatoryData/baseliiihqlaEU"
        resp = self.http_request_with_scheme_id(end_point, scheme, instruments)
        return self._convert_response_to_object(resp)

    def instrument_BASELIII_HQLA_CH(self, scheme: str, instruments: list):
        end_point = "/v1/instruments/_regulatoryData/baseliiihqlaCH"
        resp = self.http_request_with_scheme_id(end_point, scheme, instruments)
        return self._convert_response_to_object(resp)

    def instrument_SFDR(self, scheme: str, instruments: list):
        end_point = "/v1/instruments/esg/SFDRInvestee"
        resp = self.http_request_with_scheme_id(end_point, scheme, instruments)
        return self._convert_response_to_object(resp)

    def instrument_TAXONOMY(self, scheme: str, instruments: list):
        end_point = "/v1/instruments/esg/EUTaxonomyInvestee"
        resp = self.http_request_with_scheme_id(end_point, scheme, instruments)
        return self._convert_response_to_object(resp)

    def instrument_EUESGMANUFACTURER(self, scheme: str, instruments: list):
        end_point = "/v1/instruments/esg/EUESGManufacturer"
        resp = self.http_request_with_scheme_id(end_point, scheme, instruments)
        return self._convert_response_to_object(resp)

    def institution_summary(self, scheme: str, institutions: list):
        end_point = "/v1/institutions/referenceData/institutionSummary"
        resp = self.http_request_with_scheme_id(end_point, scheme, institutions)
        return self._convert_response_to_object(resp)

    def institution_symbology(self, scheme: str, institutions: list):
        end_point = "/v1/institutions/referenceData/institutionSymbology"
        resp = self.http_request_with_scheme_id(end_point, scheme, institutions)
        return self._convert_response_to_object(resp)

    def institution_SFDR(self, scheme: str, institutions: list):
        end_point = "/v1/institutions/esg/SFDRInvestee"
        resp = self.http_request_with_scheme_id(end_point, scheme, institutions)
        return self._convert_response_to_object(resp)

    def institution_TAXONOMY(self, scheme: str, institutions: list):
        end_point = "/v1/institutions/esg/EUTaxonomyInvestee"
        resp = self.http_request_with_scheme_id(end_point, scheme, institutions)
        return self._convert_response_to_object(resp)

    def market_summary(self, scheme: str, markets: list):
        end_point = "/v1/markets/referenceData/marketSummary"
        resp = self.http_request_with_scheme_id(end_point, scheme, markets)
        return self._convert_response_to_object(resp)

    def market_symboloy(self, scheme: str, markets: list):
        end_point = "/v1/markets/referenceData/marketSymbology"
        resp = self.http_request_with_scheme_id(end_point, scheme, markets)
        return self._convert_response_to_object(resp)

    def listing_EoDTimeseries(self, scheme: str, listings: list, from_date: str, to_date: str = ''):
        end_point = "/v1/listings/marketData/eodTimeseries"
        query_string = query_string = {
            'scheme': scheme,
            'ids': ",".join(listings),
            'from': from_date,
            'to': to_date
        }
        resp = self.http_request(end_point, query_string)
        return self._convert_response_to_object(resp)

    def listing_EoDTimeseries2(self, scheme: str, listings: list, from_date: str, to_date: str = ''):
        end_point = "/v1/listings/marketData/eodTimeseries"
        query_string = query_string = {
            'scheme': scheme,
            'ids': ",".join(listings),
            'from': from_date,
            'to': to_date
        }
        resp = self.http_request(end_point, query_string)
        return resp


findata = FinancialDataAPI()


def print_object_attributes(obj: object, tab_level: int = 0, min_attr_length: int = 30):
    if obj is None: return
    space_sep = "  "
    space = space_sep * tab_level

    if type(obj) == list:
        for o in obj:
            if type(o) == object or type(o) == SimpleNamespace:
                print_object_attributes(o, tab_level + 1, min_attr_length)
                print()
            else:
                print(f"{space}{o:<{min_attr_length}}")
    else:
        for attr, value in obj.__dict__.items():
            if type(value) == object or type(value) == SimpleNamespace or type(value) == list:
                print(f"{space}{attr}")

                adjusted_min_attr_length = min_attr_length - (len(space_sep) * (tab_level + 1))
                if adjusted_min_attr_length < 0: adjusted_min_attr_length = 0
                print_object_attributes(value, tab_level + 1, adjusted_min_attr_length)
            else:
                print(f"{space}{attr:<{min_attr_length}}: {value}")

#print("### ETF information")
#obj = findata.text_search("apple")
#print_object_attributes(obj, 0)

#obj = findata.instrument_EUESGMANUFACTURER("ISIN", ["CH0559601544"])
#print_object_attributes(obj, 0)

#print("### market summary")
#obj = findata.market_summary("BC", ["67"])
#print_object_attributes(obj, 0)

print("### time series")
#obj = findata.listing_EoDTimeseries("VALOR_BC", ["908440"], "1980-07-01")

#obj2 = findata.listing_EoDTimeseries2("TICKER_BC", ["AAPL_67"], "2020-07-01")
#print_object_attributes(obj, 0)

#print("### symbology")
#obj = findata.market_symboloy("BC", ["67"])
#print_object_attributes(obj, 0)

#print("### SFDR")
#obj = findata.instrument_SFDR("ISIN", ["CH0559601544"])
#print_object_attributes(obj, 0)

#obj = findata.instrument_BASELIII_HQLA_EU("ISIN", ["CH0559601544"])
#print_object_attributes(obj, 0)

#
# x = []
# y = []
# for tEntry in obj.data.listings[0].marketData.eodTimeseries:
#     if hasattr(tEntry, "close"):
#         y.append(tEntry.close)
#         x.append(tEntry.sessionDate)
#
# fig = go.Figure([go.Scatter(x=x, y=y)])
# fig.show()
#pathlib.Path('data_apple.json').write_bytes(obj2.content)

obj_bytes = pathlib.Path('data_apple.json').read_bytes()
obj_json = json.loads(obj_bytes)
objdict = obj_json['data']['listings'][0]['marketData']
d = pd.DataFrame.from_records(objdict['eodTimeseries'])
#
d = d[d['open']>0]

d.ta.ema(close='close', length=20, append=True)
d=d[10:]

# Split data into testing and training sets
#X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(d[['close']], d[['EMA_10']], test_size=.2)
# Test set
#print(X_test.describe())
#print(d)
#

# Create Regression Model
#model = LinearRegression()
# Train the model
#model.fit(X_train, y_train)
# Use model to make predictions
#y_pred = model.predict(X_test)

# Printout relevant metrics
#print("Model Coefficients:", model.coef_)
#print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
#print("Coefficient of Determination:", r2_score(y_test, y_pred))

#print(y_pred)

fig = go.Figure([
#     # go.Candlestick(
#     # x=d.sessionDate,
#     #     open=d.open,
#     #     high=d.high,
#     #     low=d.low,
#     #     #close=d.close
#     # ),
     go.Scatter(x=d['sessionDate'], y=d['open']),
     #go.Scatter(x=d['sessionDate'], y=d['open'].rolling(50).mean()),
     go.Scatter(x=d['sessionDate'], y=d.EMA_20)
 ])
fig.show()
#
# #print(d)

