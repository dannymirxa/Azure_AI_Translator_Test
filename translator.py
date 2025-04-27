import numpy as np
import polars as pl
import requests, uuid, json
from typing import List, Tuple, Optional
import os
from dotenv import load_dotenv

load_dotenv('.env')

# Add your key and endpoint
key = os.getenv("AZURE_TEXT_TRANSLATION_KEY")
endpoint = os.getenv("AZURE_TEXT_TRANSLATION_ENDPOINT")

location = "eastus"
path = "/translate"
constructed_url = endpoint + path

class Translator():
    def __init__(self, input: str, output: str):
        self.input = input
        self.output = output
    
    def create_df(self) -> pl.DataFrame:
        df = pl.read_csv(self.input)
        return df

    def translate_text(self, text: str, translate_to_language: List[str] = ['en']) -> Tuple[str|None, List[int]]:
        if text is pl.Null|None:
            return None, [0]
            
        params = {
            'api-version': '3.0',
            'to': translate_to_language, # Translate to English
            'includeSentenceLength': True
        }

        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        # Create the body with the text to translate
        body = [{'text': text}]

        # Make the request to the Translator API
        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        response = request.json()

        # Extract the translated text
        try:
            translated_text = response[0]['translations'][0]['text']
            translated_text_length = response[0]['translations'][0]['sentLen']['transSentLen']
        except (IndexError, KeyError):
            translated_text = None  # Handle errors gracefully
            translated_text_length = [0]

        return translated_text, translated_text_length
    
    def split_text_by_lengths(self, text: str, lengths: list) -> List[str]:
        sentences = []
        start = 0
        for length in lengths:
            # Slice the string from the current start index to the next "start+length"
            sentences.append(text[start:start+length])
            start += length
        return sentences
    
    
if __name__=="__main__":
    translator = Translator(input= "./text-zh.csv", output= "./text-zh_translated.csv")
    df = translator.create_df()
    
    # translated_text, translated_text_length = translator.translate_text(text= 'Baker&Spice是上海Wagas（沃歌斯，后文为方便起见均用该名称代指Baker&Spice）旗下的餐厅。沃歌斯青岛首家餐厅位于万象城LG层中门外侧，意外不是很起眼的位置，在猪头肉印象里这家餐厅也相对比较低调，一直不温不火。正如其名，该餐厅的主营的糕点和意面披萨等简餐都有非常不错的水准。近期经常来沃歌斯，所以就做一下较深度点评。\\n\\n       沃歌斯的装修陈设简洁明亮，偏向北欧范儿。而没做吊顶的裸顶和垂下来的白炽灯又带着一丝工业风。高端文艺范的环境已经成功拉升了餐厅的格调。\\n        在沃歌斯点过两次沙拉，一次凯撒沙拉一次鸡肉芒果沙拉。虽然猪头肉并不喜欢吃生菜多的沙拉，不过沃歌斯的沙拉味道确实非常不错。凯撒沙拉是经典名菜了。沃歌斯的凯撒沙拉默认是不加鸡肉，加鸡肉需要加18元。猪头肉就后悔当时没有加钱加鸡肉。简版的凯撒沙拉是鹌鹑蛋、肉脯碎配蔬菜，上面撒着类似面包糠的碎屑。沃歌斯的沙拉首先胜在食材质量，鹌鹑蛋和生菜不说，肉脯碎的味道就很正，大有双鱼牌的赶脚。另外凯撒沙拉的的酱汁也非常地道。鸡肉芒果沙拉是这里沙拉类的销售冠军。鸡肉烤的火候适中，肉质紧实。芒果切的非常大块，吃到确实很爽。以牛油果为首的各类蔬果配菜也非常丰富。特制的酸甜味酱汁配奶油的味道清新独特，非常开胃。\\n\\n        意面半两尝过三款，各有特色，总体感觉都非常不错。细面配三文鱼通过烹制过程中加入少量奶油很好的烘托了三文鱼浓郁的味道。而选择在食用过程中挤一点柠檬汁则可以中和油腻感和厚重感，增加一丝清新。白葡萄酒海鲜意面是意面类的销售冠军，辅料是虾仁、比管、花蛤还有一点圣女果干和西芹。这款意面比其他意面都要少油，相对的却可以明显的尝出白葡萄酒的清香，而圣女果干沉淀的酸甜和西芹的水嫩很好的配合了白葡萄酒的香味。使用的海鲜鲜度也非常高，虾仁出自本地鲜活的海捕虾。可以明显尝得出来整道意面并没有使用提鲜的调味料，鲜度完全来自于以海虾为首的海鲜。绿酱鸡肉意面使用的罗勒酱味道非常纯正，单是罗勒酱就很赞了。配上奶油口感浓醇爽滑，搭配的空心粉和鸡肉也超合拍。这道意面强力推荐。\\n\\n        沃歌斯还有几款东南亚风的咖喱饭，猪头肉点过其中一款泰式辣味牛肉。这款咖喱是由牛肉、泡发海菜、菜椒和腰果烹制的。牛肉选用的是类似里脊牛腩的部位，肉质很嫩又有嚼劲。咖喱的香辛料绘制出的鲜辣味道很过瘾，而海菜的谷氨酸做了出色的点缀。咖喱的汤汁比较浓稠，配菜的长粒香米质量也很好。配在一起整道菜的体验还是很不错的。\\n\\n         披萨试过两款，意式海鲜披萨和火腿蘑菇披萨。沃歌斯的披萨都是薄饼底。可能是在面团里放了小苏打，披萨的饼底非常蓬松酥脆。披萨的芝士也是超级多，味道也很正，跟特制的薄饼底正是绝配。意式海鲜披萨使用的海鲜与海鲜意面相同，也是突出了以海虾为首的海鲜带来的鲜味。而火腿蘑菇披萨则以口菇特有的鲜味很好的衬托了优质火腿的肉香。这两款披萨都非常推荐。\\n\\n         沃歌斯的甜品品质也非常不错。树莓巧克力挞的酱汁酸味很重，是纯正的树莓果酱。芒果杏仁芝士蛋糕的芒果切得也很大块，中间夹的应该是奶油和芝士混合的酱，饼底的杏仁蛋糕味道很正。胡萝卜蛋糕的酱类似卡仕达酱，但很甜，就像混合了蜂蜜或者炼乳，口感非常醇厚。胡萝卜蛋糕的饼底口感也是醇厚到类似粗粮蛋糕或是枣糕的感觉，里面有胡萝卜和山核桃，也是非常不错。\\n\\n        沃歌斯的简餐和甜品水平都比较高，在同等价位里算是制作非常精致味道比较考究的了。面包没尝过，不妄作评断。目前不是很火爆，所以环境还是很好的。缺点是临街店内会有飞虫。另外猪头肉爆料一下，楼上CGV影院的会员卡在这里除了饮品之外都可享七五折，在万象城性价比还是蛮高的。', translate_to_language= ['en'])
    # translated_text, translated_text_length = translator.translate_text(text= pl.Null, translate_to_language= ['en'])
    # print(translated_text)
    # print(translated_text_length)