from django.db import models
# from django.contrib.auth.models import User

# Create your models here.
# class Conversation(models.Model):  
#     user = models.ForeignKey(User, on_delete=models.CASCADE)  
#     start_time = models.DateTimeField(auto_now_add=True)  
#     end_time = models.DateTimeField(null=True, blank=True)

# class Message(models.Model):  
#     user = models.ForeignKey(User, on_delete=models.CASCADE)  
#     conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)  
#     text = models.TextField()  
#     timestamp = models.DateTimeField(auto_now_add=True)

class Users(models.Model):
    email_id = models.CharField(max_length=50)
    # name = models.CharField(max_length=10)
    # password = models.CharField(max_length=20)
    isAdmin =  models.BooleanField(default=False)

    def __str__(self):
        return self.email_id+self.isAdmin

class Conversation(models.Model):
    email_id = models.CharField(max_length=50)
    # conv_id = models.IntegerField(default=1, unique=True)
    conv_name = models.CharField(max_length=50)
    pinned = models.BooleanField(default=False)
    date = models.DateField(auto_now_add=True, null=True)

    def __str__(self):
        return self.email_id+self.conv_name

class Message(models.Model):
    # msg_id = models.IntegerField(default=1, unique=True)
    msg = models.TextField()
    conv_id = models.IntegerField() 
    msg_type = models.CharField(max_length=20)
    response_from = models.TextField(default="not applicable")
    feedback = models.CharField(max_length=10, default='neutral')
    time = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.msg+self.conv_id+self.msg_type+self.time

class UserDocuments(models.Model):
    # udoc_id = models.IntegerField(default=1, unique=True)
    udoc_path = models.CharField(max_length=100)
    email_id =  models.CharField(max_length=50)

class EnterpriseDocuments(models.Model):
    # edoc_id = models.IntegerField(default=1, unique=True)
    edoc_path = models.CharField(max_length=100)

class EnterpriseDictionary(models.Model):
    original_word = models.TextField()
    enterprise_word = models.TextField()

class RephrasedQuestions(models.Model):
    original_query = models.TextField()  
    rephrased_query = models.TextField()

class SystemPrompt(models.Model):
    prompt = models.TextField()

class LLMTemperature(models.Model):
    temperature = models.DecimalField(max_digits=2, decimal_places=1)

class LLM(models.Model):
    name = models.TextField() 
    endpoint = models.TextField()
    key = models.TextField()
    enabled = models.BooleanField(default=False)

    def save(self, *args, **kwargs):  
        if not self.pk and not LLM.objects.exists():  
            self.enabled = True  
        super(LLM, self).save(*args, **kwargs)

class ContentFilters(models.Model):
    hate = models.IntegerField()
    sexual = models.IntegerField()
    violence = models.IntegerField()
    self_harm = models.IntegerField()