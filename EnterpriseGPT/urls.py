"""
URL configuration for EnterpriseGPT project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
# from RAG.views import RagResponse,DocumentUploadView
from RAG import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('rag/response/', RagResponse.as_view(), name='ragresponse'),
    # path('rag/upload/', DocumentUploadView.as_view(), name='upload_document')
    path('rag/response/', views.RagResponse),
    path('rag/upload/', views.DocumentUploadView),
    path('rag/userUpload/<str:email_id>', views.userDocumetUpload),
    path('rag/userDelete/<int:id>', views.deleteUserDoc),
    path('rag/deleteSpecificUserDocs/<str:email_id>', views.deleteSpecificUserDocs),
    path('rag/deleteAlluserDocs/', views.deleteAllUserDocs),
    path('rag/getUserDocs/', views.getUserDocs),
    path('rag/userRagResponse/', views.userDocResponse),
    path('rag/delete/<int:id>', views.deleteDocument),
    path('delete_all/', views.delete_all_enterprisedocs),
    path('users/', views.UserDetails),
    path('conv/', views.ConvDetails),
    path('updateconv/<int:id>',views.ConvDetailsPK),
    path('allConv/<str:email>/', views.getAllUserConv),
    path('conv/msgs/<int:conv_id>/', views.getChatmsgs),
    path('conv/msgs/update/<int:id>', views.updateMsg),
    path('dummy/', views.dummyApiCall),
    path('addEnterpriseWord/', views.addEnterpriseWord),
    path('getAllWords/', views.getAllEnterpriseWords),
    path('updateWord/<int:id>', views.updateEnterpriseWords),
    path('deleteWord/<int:id>', views.deleteEnterpriseWord),
    path('addPrompt/', views.addSystemPrompt),
    path('getPrompt/<int:id>', views.getSystemPrompt),
    path('updatePrompt/<int:id>', views.updateSystemPrompt),
    path('addTemp/', views.addTemperature),
    path('updateTemp/<int:id>', views.updateTemp),
    path('getTemp/<int:id>', views.getTemp),
    path('getDashboardData/', views.dashBoardData),
    path('addLLM/', views.addLLM),
    path('llm/list/', views.getAllLLMs),
    path('llm/enabled/list/', views.getEnabledLLMs),
    path('llm/update/<int:id>', views.updateLLM),
    path('llm/delete/<int:id>', views.deleteLLM),
    path('llm/contentFilters/add', views.addContentFilter),
    path('llm/contentFilter/update/<int:id>', views.updateContentFilter),
    path('llm/contentFilter/<int:id>', views.getContentFilter)
]
