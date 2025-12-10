import os
import pandas as pd
from django.core.management.base import BaseCommand
from healthcare.models import Article, QuestionAnswer


class Command(BaseCommand):
    help = 'Import dữ liệu từ CSV files (bloomax.csv và filtered-question-answers.csv)'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('='*50))
        self.stdout.write(self.style.SUCCESS('IMPORT DỮ LIỆU Y TẾ'))
        self.stdout.write(self.style.SUCCESS('='*50 + '\n'))

        # Đường dẫn file CSV - nằm ở thư mục cha của Healthcare_Recommendation_System
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        parent_dir = os.path.dirname(base_dir)  # Thư mục cha của Website
        
        articles_csv = os.path.join(parent_dir, 'bloomax.csv')
        qa_csv = os.path.join(parent_dir, 'filtered-question-answers.csv')

        # Import bài viết
        if os.path.exists(articles_csv):
            self.import_articles(articles_csv)
        else:
            self.stdout.write(self.style.WARNING(f'⚠ Không tìm thấy file: {articles_csv}\n'))

        # Import câu hỏi-trả lời
        if os.path.exists(qa_csv):
            self.import_qa(qa_csv)
        else:
            self.stdout.write(self.style.WARNING(f'⚠ Không tìm thấy file: {qa_csv}\n'))

        # Thống kê
        total_articles = Article.objects.count()
        total_qas = QuestionAnswer.objects.count()

        self.stdout.write(self.style.SUCCESS('='*50))
        self.stdout.write(self.style.SUCCESS('THỐNG KÊ CUỐI CÙNG'))
        self.stdout.write(self.style.SUCCESS('='*50))
        self.stdout.write(self.style.SUCCESS(f'Tổng bài viết: {total_articles}'))
        self.stdout.write(self.style.SUCCESS(f'Tổng câu hỏi-trả lời: {total_qas}'))
        self.stdout.write(self.style.SUCCESS('='*50))

    def import_articles(self, csv_path):
        """Import bài viết từ bloomax.csv"""
        self.stdout.write(f'Đang import bài viết từ {csv_path}...')

        df = pd.read_csv(csv_path, encoding='utf-8')

        imported_count = 0
        skipped_count = 0

        for idx, row in df.iterrows():
            try:
                article, created = Article.objects.get_or_create(
                    link=row['link'],
                    defaults={
                        'title': row['title'],
                        'content': row['txt']
                    }
                )

                if created:
                    imported_count += 1
                    if (imported_count % 10) == 0:
                        self.stdout.write(f'  ✓ Đã import {imported_count} bài viết')
                else:
                    skipped_count += 1
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'  ✗ Lỗi import hàng {idx+2}: {e}'))

        self.stdout.write(self.style.SUCCESS(f'\nKết quả import bài viết:'))
        self.stdout.write(self.style.SUCCESS(f'  - Mới: {imported_count}'))
        self.stdout.write(self.style.SUCCESS(f'  - Trùng: {skipped_count}'))
        self.stdout.write(self.style.SUCCESS(f'  - Tổng: {imported_count + skipped_count}\n'))

    def import_qa(self, csv_path):
        """Import câu hỏi-trả lời từ filtered-question-answers.csv"""
        self.stdout.write(f'Đang import câu hỏi-trả lời từ {csv_path}...')

        df = pd.read_csv(csv_path, encoding='utf-8')

        imported_count = 0
        skipped_count = 0

        for idx, row in df.iterrows():
            try:
                # Chuẩn hóa topic
                topic = row['topic'].lower().strip() if pd.notna(row['topic']) else 'y tế chung'
                
                qa, created = QuestionAnswer.objects.get_or_create(
                    qa_id=row['id'],
                    defaults={
                        'question': row['question'],
                        'answer': row['answer'],
                        'topic': topic
                    }
                )

                if created:
                    imported_count += 1
                    if (imported_count % 10) == 0:
                        self.stdout.write(f'  ✓ Đã import {imported_count} câu hỏi')
                else:
                    skipped_count += 1
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'  ✗ Lỗi import hàng {idx+2}: {e}'))

        self.stdout.write(self.style.SUCCESS(f'\nKết quả import câu hỏi-trả lời:'))
        self.stdout.write(self.style.SUCCESS(f'  - Mới: {imported_count}'))
        self.stdout.write(self.style.SUCCESS(f'  - Trùng: {skipped_count}'))
        self.stdout.write(self.style.SUCCESS(f'  - Tổng: {imported_count + skipped_count}\n'))
