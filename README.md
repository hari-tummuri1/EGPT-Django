Always Use **main** branch for installation

**Installation process**
  1. git clone https://github.com/hari-tummuri1/EGPT-Django.git
  2. git checkout main
  3. Get API keys from gemini, Pinecone
  4. create .env file in EnterpriseGPT folder and add apiKeys with below variable names GOOGLE_API_KEY, PINECONE_API_KEY

**Creating Python Virtual environment**
  1. Install Python 3.11 if you haven't already. You can download it from [Python's official website.](https://www.python.org/downloads/)
  2. **Create the virtual environment**: python -m venv myenv
  3. **Activate the environment:**
       1. On Windows:  myenv\Scripts\activate
       2. On macOS/Linux: source myenv/bin/activate
  4. **Verify the Python version:**  python --version (It should show Python 3.11.x.)

**Commands to run the server**

  1. pip install -r reequirements.txt
  2. python manage.py runserver
  3. ngrok config add-authtoken 2d8t3Bfhtrpwbwhew7JQIT62wmk_7ZwyHUBSoYChbhnb18RHh
  4. ngrok http --domain informed-hardly-oyster.ngrok-free.app 8000
