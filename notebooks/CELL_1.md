
## Cell 1: Setup GitHub Authentication

```python
# Setup GitHub authentication
import os
from getpass import getpass

# Get GitHub token (paste it when prompted)
github_token = getpass('Enter your GitHub personal access token: ')
github_username = input('Enter your GitHub username (adantas): ') or 'adantas'
github_repo = 'aquascan-gnns'

# Store for later use
os.environ['GITHUB_TOKEN'] = github_token
os.environ['GITHUB_USERNAME'] = github_username
os.environ['GITHUB_REPO'] = github_repo

print("✅ GitHub credentials stored!")
```