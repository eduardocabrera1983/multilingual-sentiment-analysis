# Environment Configuration Guide

This guide explains how to securely configure environment variables for the Multilingual Sentiment Analysis application.

## 🔐 Security Overview

This application uses environment variables to manage sensitive configuration data, ensuring that credentials and secrets are never hardcoded in the source code.

## 📋 Setup Instructions

### 1. Copy Environment Template

Copy the provided template to create your own environment file:

```bash
cp .env.template .env
```

### 2. Edit Environment Variables

Open the `.env` file and update the values according to your setup:

```bash
# Required for production
SECRET_KEY=your-actual-secret-key-here

# Optional but recommended
HF_TOKEN=your-huggingface-token-here
```

### 3. Generate Secure Secret Key

For the `SECRET_KEY`, generate a cryptographically secure random key:

```python
# Run this Python command to generate a secure key
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the generated key and paste it as your `SECRET_KEY` value.

## 🔑 Key Environment Variables

### Required for Production

- **`SECRET_KEY`**: Flask session encryption key (REQUIRED for production)

### Optional Configuration

- **`HF_TOKEN`**: Hugging Face API token for private models or increased rate limits
- **`FORCE_CPU`**: Set to `true` to force CPU-only mode
- **`FORCE_GPU`**: Set to `true` to force GPU mode
- **`FLASK_DEBUG`**: Set to `true` for development debugging

### Model Configuration

- **`SENTIMENT_MODEL_1`**: Primary sentiment model ID (default: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual)
- **`SENTIMENT_MODEL_2`**: Secondary sentiment model ID (default: cardiffnlp/twitter-roberta-base-sentiment-latest)
- **`ASPECT_MODEL`**: Aspect classification model ID (default: facebook/bart-large-mnli)

## 🛡️ Security Best Practices

### 1. Never Commit .env Files
The `.env` file is automatically ignored by Git to prevent accidental commits. **Never** commit this file to version control.

### 2. Use Strong Secret Keys
- Generate cryptographically secure random keys
- Use at least 32 characters for the SECRET_KEY
- Rotate keys regularly in production

### 3. Limit Environment Access
- Only provide necessary environment variables
- Use different .env files for different environments (dev/staging/prod)
- Store production secrets in secure key management systems

### 4. Validate Environment Variables
The application will warn you if required environment variables are missing and will use secure defaults where possible.

## 🌍 Environment-Specific Configurations

### Development (.env.development)
```bash
FLASK_DEBUG=true
SECRET_KEY=dev-key-not-for-production
LOG_LEVEL=DEBUG
```

### Production (.env.production)
```bash
FLASK_DEBUG=false
SECRET_KEY=your-very-secure-production-key
LOG_LEVEL=INFO
FORCE_HTTPS=true
```

### Docker (.env.docker)
```bash
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
DOCKER_PORT=5000
```

## 📖 Hugging Face Token Setup

If you want to use private models or avoid rate limits:

1. Create account at [https://huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create a new token with `read` permissions
4. Add to your `.env` file:
   ```bash
   HF_TOKEN=hf_your_token_here
   ```

## 🚨 Troubleshooting

### Missing SECRET_KEY Warning
If you see: `[WARNING] No SECRET_KEY found in environment`
- Add `SECRET_KEY=your-key-here` to your `.env` file
- Generate a new key using the Python command above

### Model Loading Issues
If models fail to load:
- Check your `HF_TOKEN` if using private models
- Verify model IDs in environment variables
- Check internet connectivity for model downloads

### Environment Not Loading
If variables aren't being read:
- Ensure `.env` file is in the project root directory
- Check file permissions (should be readable)
- Verify `python-dotenv` is installed: `pip install python-dotenv`

## 📁 File Structure

```
project-root/
├── .env                 # Your environment variables (DO NOT COMMIT)
├── .env.template        # Template file (safe to commit)
├── .gitignore          # Includes .env files
└── app.py              # Loads environment variables
```

## 🔗 Related Documentation

- [Flask Configuration](https://flask.palletsprojects.com/en/2.0.x/config/)
- [Python-dotenv Documentation](https://github.com/theskumar/python-dotenv)
- [Hugging Face Tokens](https://huggingface.co/docs/hub/security-tokens)
- [Environment Variables Best Practices](https://12factor.net/config)