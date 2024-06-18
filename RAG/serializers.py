from rest_framework import serializers  
from .models import Users, Conversation, Message, UserDocuments, EnterpriseDocuments, EnterpriseDictionary, RephrasedQuestions, SystemPrompt, LLMTemperature, LLM, ContentFilters
  
# class MessageSerializer(serializers.ModelSerializer):  
#     class Meta:  
#         model = Message  
#         fields = ['user', 'text', 'timestamp']  
  
# class ConversationSerializer(serializers.ModelSerializer):  
#     messages = MessageSerializer(many=True, read_only=True)  
  
#     class Meta:  
#         model = Conversation  
#         fields = ['id', 'user', 'start_time', 'end_time', 'messages'] 

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Users
        fields = ['email_id', 'isAdmin'] 

class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ['id','email_id', 'conv_name','pinned','date']

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id','msg', 'conv_id', 'msg_type', 'feedback','time', 'response_from']

class UserDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserDocuments
        fields = ['id','udoc_path', 'email_id']

class EnterpriseDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = EnterpriseDocuments
        fields = ['id','edoc_path']

class EnterpriseDictionarySerializer(serializers.ModelSerializer):
    class Meta:
        model = EnterpriseDictionary
        fields = ['id', 'original_word', 'enterprise_word']

class RephrasedQuestionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = RephrasedQuestions
        fields = ['id', 'original_query', 'rephrased_query']

class SystemPromptSerializer(serializers.ModelSerializer):
    class Meta:
        model = SystemPrompt
        fields = ['id', 'prompt']

class LLMTemperatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = LLMTemperature
        fields = ['id', 'temperature']

class LLMSerializer(serializers.ModelSerializer):
    class Meta:
        model = LLM
        fields = ['id', 'name', 'endpoint' , 'key', 'enabled']

    def create(self, validated_data):  
        instance = LLM(**validated_data)  
        instance.save()  
        return instance
    
    def update(self, instance, validated_data):  
        if 'enabled' in validated_data and validated_data['enabled'] is False:  
            if LLM.objects.filter(enabled=True).count() <= 1:  
                raise serializers.ValidationError({"error" : "At least one instance should be enabled"})  
        return super().update(instance, validated_data) 

class ContentFilterSerializer(serializers.ModelSerializer):
    class Meta:
        model = ContentFilters
        fields = ['id', 'hate', 'sexual', 'violence', 'self_harm']

