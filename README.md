Use main branch

**Installation process**

step 1:- Make sure you have python 3.11
step 2:- git clone https://github.com/hari-tummuri1/EGPT-Django.git
step 3:- git checkout main
step 4:- Get API keys from gemini, Pinecone
step 5:- create .env file in EnterpriseGPT folder and add apiKeys with below variable names GOOGLE_API_KEY, PINECONE_API_KEY



**Commands to run the server**

pip install -r reequirements.txt

python manage.py runserver

ngrok config add-authtoken 2d8t3Bfhtrpwbwhew7JQIT62wmk_7ZwyHUBSoYChbhnb18RHh

ngrok http --domain informed-hardly-oyster.ngrok-free.app 8000
