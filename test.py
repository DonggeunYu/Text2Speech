#-*- coding: utf-8 -*-

from tacotron.utils.symbols import symbols

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

a = '한글'
print repr(list(a)).decode('string-escape')

