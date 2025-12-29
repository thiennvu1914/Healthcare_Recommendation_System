from django.core.management.base import BaseCommand
from chatbot.models import Article, QA
import pandas as pd
from pathlib import Path


class Command(BaseCommand):
    help = 'Import QAs and Articles from CSV files'

    def handle(self, *args, **options):
        # Path to data directory
        base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        data_dir = base_dir / 'data'
        
        qa_file = data_dir / 'QAs.csv'
        articles_file = data_dir / 'articles.csv'
        
        # Import Q&A
        if qa_file.exists():
            self.stdout.write('Importing Q&A...')
            df_qa = pd.read_csv(qa_file)
            
            # Clear existing data
            QA.objects.all().delete()
            
            batch = []
            for idx, row in df_qa.iterrows():
                batch.append(QA(
                    qa_id=f"qa_{idx}",
                    question=str(row.get('question', ''))[:5000],
                    answer=str(row.get('answer', ''))[:10000],
                    topic=str(row.get('topic', 'Khác'))[:200],
                    topic_original=str(row.get('topic_original', ''))[:200],
                ))
                
                if len(batch) >= 1000:
                    QA.objects.bulk_create(batch, ignore_conflicts=True)
                    self.stdout.write(f'  Imported {idx + 1} Q&A...')
                    batch = []
            
            if batch:
                QA.objects.bulk_create(batch, ignore_conflicts=True)
            
            self.stdout.write(self.style.SUCCESS(f'Successfully imported {df_qa.shape[0]} Q&A'))
        else:
            self.stdout.write(self.style.WARNING(f'Q&A file not found: {qa_file}'))
        
        # Import Articles
        if articles_file.exists():
            self.stdout.write('Importing Articles...')
            df_articles = pd.read_csv(articles_file)
            
            # Clear existing data
            Article.objects.all().delete()
            
            batch = []
            for idx, row in df_articles.iterrows():
                batch.append(Article(
                    article_id=str(row.get('id', f'art_{idx}'))[:50],
                    title=str(row.get('title', ''))[:500],
                    text=str(row.get('text', ''))[:20000],
                ))
                
                if len(batch) >= 1000:
                    Article.objects.bulk_create(batch, ignore_conflicts=True)
                    self.stdout.write(f'  Imported {idx + 1} articles...')
                    batch = []
            
            if batch:
                Article.objects.bulk_create(batch, ignore_conflicts=True)
            
            self.stdout.write(self.style.SUCCESS(f'Successfully imported {df_articles.shape[0]} articles'))
        else:
            self.stdout.write(self.style.WARNING(f'Articles file not found: {articles_file}'))
        
        self.stdout.write(self.style.SUCCESS('Import completed!'))
