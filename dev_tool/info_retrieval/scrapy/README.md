# Scrapy  : Web Crawler

## Preliminary
**Learning Material**\
https://segmentfault.com/a/1190000013178839 \
**API**\
https://docs.scrapy.org/en/latest/index.html \
**Project Structure** :
```text
scrapy.cfg                - Deploy the configuration file
project_name/             - Name of the project
   _init_.py
   items.py               - It is project's items file
   pipelines.py           - It is project's pipelines file
   settings.py            - It is project's settings file
   spiders                - It is the spiders directory
      _init_.py
      spider_name.py
```
**Architecture**\
![architecture](https://segmentfault.com/img/bVco7P)
- engine : communication
- scheduler : receive request, arrange request
- downloader : download from internet
- spider : process response and get data, return item and url
- item pipeline : postprocessing
- downloader middlewares : extension for download function
- spider middlewares : extension for communication between engine and spider

**Design Step**
1. create project : `scrapy startproject xxx`
2. specify goals : `items.py`
3. make spider : `scrapy genspider name url` `xxspider.py`
4. first crawl : `scrapy crawl name`, check obtained response.body (better in .html).
5. parsing : `def parse(self, response)`
6. postprocess item : `pipelines.py`
7. crawl and store : `scrapy crawl name -o path.json`

## Command Line
**create project**
```commandline
scrapy startproject my_scrapy
cd my_scrapy
```
**generate spider**
```commandline
scrapy genspider quotes quotes.toscrape.com
```
**command list**
```commandline
scrapy -h
```
**crawl**
```commandline
scrapy crawl quotes -o result.json
```
`quotes is spider name`
`json, jsonl, csv, xml`


