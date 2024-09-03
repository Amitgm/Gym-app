FROM python:3.10

# Set the working directory
WORKDIR /code

# Add requirements file to the Docker image
COPY ./requirements.txt /code/requirements.txt

# Install Python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Add the app folder into the Docker image
COPY ./app/ /code/app/
# COPY ./app/.streamlit /code/app/.streamlit

WORKDIR /code/app

# Set environment variable for Flask
# ENV FLASK_APP=app/main.py
# ENV FLASK_RUN_PORT=80

# Specify default command to run the Flask app
# CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]

# Set the environment variable for Streamlit to find the config file
# ENV STREAMLIT_CONFIG=/code/app/.streamlit/config.toml

ENV STREAMLIT_APP=app/main.py


# Run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=80", "--server.address=0.0.0.0"]

# CMD ["gunicorn", "--bind", "0.0.0.0:80", "app.main:app"]