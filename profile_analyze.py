import pstats
from pstats import SortKey
p = pstats.Stats('main_stats')
p.sort_stats(SortKey.CUMULATIVE).print_stats(100)