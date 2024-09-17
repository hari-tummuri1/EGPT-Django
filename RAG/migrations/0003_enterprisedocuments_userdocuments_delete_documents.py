# Generated by Django 5.0.4 on 2024-05-20 11:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RAG', '0002_documents'),
    ]

    operations = [
        migrations.CreateModel(
            name='EnterpriseDocuments',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('edoc_id', models.IntegerField(default=1, unique=True)),
                ('edoc_path', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='UserDocuments',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('udoc_id', models.IntegerField(default=1, unique=True)),
                ('udoc_path', models.CharField(max_length=100)),
                ('email_id', models.CharField(max_length=20)),
            ],
        ),
        migrations.DeleteModel(
            name='Documents',
        ),
    ]
