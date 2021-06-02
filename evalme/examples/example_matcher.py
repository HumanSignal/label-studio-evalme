from evalme.matcher import Matcher

loader = Matcher(url="http://127.0.0.1:8000",
                 token="ACCESS_TOKEN",
                 project='1')
loader.refresh()

loader = Matcher()
loader.load('your_filename')