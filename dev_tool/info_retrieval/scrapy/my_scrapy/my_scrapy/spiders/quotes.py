import scrapy
from misc.scrapy.my_scrapy.my_scrapy.items import QuotesItem

class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    allowed_domains = ['quotes.toscrape.com']
    start_urls = ['https://quotes.toscrape.com/']

    def parse(self, response):
        # after have gotten response, how we parse it
        # how we get url and item

        'first, we have a look at what it looks like :'
        # filename = 'result.html'
        # with open(filename, 'w') as f:
        #     f.write(str(response.body))

        'then we get the items and url'
        # items = []
        # for each in response.xpath('//div[@class="quote"]'):
        #     item = QuotesItem()
        #     text = each.xpath('./span[@class="text"]/text()').extract_first()
        #     author = each.xpath('.//small[@class="author"]/text()').extract_first(),
        #     tags = each.xpath('.//div[@class="tags"]/a[@class="tag"]/text()').extract()
        #
        #     # xpath returns a list with one element
        #     item['author'] = author[0]
        #     item['text'] = text[0]
        #     item['tags'] = tags[0]
        #     items.append(item)
        #
        # next_page_url = response.xpath('//li[@class="next"]/a/@href').extract_first()
        # next_page_request = None
        # if next_page_url is not None:
        #     next_page_request = scrapy.Request(response.urljoin(next_page_url))
        # return items, next_page_request

        'we can also treat this as iterator/generator : https://www.runoob.com/w3cnote/python-yield-used-analysis.html'
        for quote in response.xpath('//div[@class="quote"]'):
            yield {
                'text': quote.xpath('./span[@class="text"]/text()').extract_first(),
                'author': quote.xpath('.//small[@class="author"]/text()').extract_first(),
                'tags': quote.xpath('.//div[@class="tags"]/a[@class="tag"]/text()').extract()
            }

        next_page_url = response.xpath('//li[@class="next"]/a/@href').extract_first()
        if next_page_url is not None:
            yield scrapy.Request(response.urljoin(next_page_url))







