from ohmysportsfeedspy import MySportsFeeds

msf = MySportsFeeds(version="1.0")

msf.authenticate("jbalawej", "Lsin00")

output = msf.msf_get_data(league='nba',season='2016-2017-regular',feed='cumulative_player_stats',format='csv')

