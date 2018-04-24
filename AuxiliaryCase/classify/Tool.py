# -*- coding: utf-8 -*-

import re
import string
class Tool:
    def __init__(self):
        self.removeDiv = re.compile(r'<div.*?></div>')
        self.removeImg = re.compile(r'<img.*?>| {7}|')
        self.removeAddr = re.compile(r'<a.*?>|</a>')
        self.replaceLine = re.compile(r'<tr>|<div>|</div>|</p>')
        self.replaceTD = re.compile(r'<td>')
        self.replacePara = re.compile(r'<p.*?>')
        self.replaceBR = re.compile(r'<br><br>|<br>')
        self.removeExtraTag = re.compile('<.*?>')
        self.removeSpace = re.compile(r'\s+')
        self.removeNbsp = re.compile(r'&nbsp;')
        self.removeWord = re.compile(u'[^a-zA-Z\-_\u4e00-\u9fa5\s]+')

    def replace(self, x):
        x = re.sub(self.removeImg, "", x)
        x = re.sub(self.removeAddr, "", x)
        x = re.sub(self.replaceLine, "", x)
        x = re.sub(self.replaceTD, "", x)
        x = re.sub(self.replacePara, "", x)
        x = re.sub(self.replaceBR, "", x)
        x = re.sub(self.removeDiv, "", x)
        x = re.sub(self.removeExtraTag, "", x)
        x = re.sub(self.removeNbsp,"", x)
        return x.strip()

    def respace(self,x):
        x = re.sub(self.removeSpace," ",x.strip())
        return x.strip()

    def reSpecialCharacters(self,x):
        x = re.sub(self.removeWord," ",x)
        return x.strip()

    def clean_html_js(self,html):
        cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())

        cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)

        cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)

        cleaned = re.sub(r"&nbsp;", " ", cleaned)

        cleaned = re.sub(r"  ", " ", cleaned)

        cleaned = re.sub(r"  ", " ", cleaned)
        return cleaned.strip()

    def clean_url(self,html):
        cleaned = re.sub(r"((ht|f)tps?):\/\/([\w\-]+(\.[\w\-]+)*\/)*[\w\-]+(\.[\w\-]+)*\/?(\?([\w\-\.,@?^=%&:\/~\+#]*)+)?","",html.strip())
        return  cleaned.strip()

    def clean_punctuation(self,html):
        cleaned = re.sub(re.escape(string.punctuation)," ",html)
        return  cleaned.strip()