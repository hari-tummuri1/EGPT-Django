# Generated by Django 5.0.4 on 2024-05-21 04:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RAG', '0004_remove_conversation_conv_id_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='users',
            name='email_id',
            field=models.CharField(max_length=50),
        ),
    ]
