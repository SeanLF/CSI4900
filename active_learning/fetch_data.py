import newspaper
from newspaper import Article as NArticle
import progressbar
import feedparser
from .models import Article


class FetchData:
    queries = ['cyber security', 'security', 'privacy', 'cyber', 'health', 'sport', 'trump']

    # Fetch Google News URLs for queries
    def getGoogleNewsURLs(self):
        links_per_query = {}
        print('Fetching links from Google News')
        bar = progressbar.ProgressBar()
        for query in bar(self.queries):
            queryURL = "https://news.google.ca/news/section?q=" + query.replace(' ', '+') + "&output=atom"
            links_per_query[query] = []
            for entry in feedparser.parse(queryURL)['entries']:
                links_per_query[query].append(entry['link'])

        return links_per_query

    def processAndSaveDataFromLinks(self, links_per_query):
        print("\n\nMining top articles from", len(self.queries), "Google News searches.")

        for query, urls in links_per_query.items():
            print("\nQuery:", query)
            # Display progress
            bar = progressbar.ProgressBar()

            for url in bar(urls):
                # Fetch the HTML from the links
                a = NArticle(url=url, language="en")
                a.download()

                # Parse and extract title, text and keywords
                try:
                    a.parse()
                    a.nlp()
                except newspaper.article.ArticleException:
                    print('Failed for', url)
                    pass

                # Save data in database
                article = Article(query=query, url=url, title=a.title, text=a.text, keywords=a.keywords)
                article.save()
