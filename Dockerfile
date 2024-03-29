FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -i https://pypi.org/simple -r requirements.txt

# Copy the apimodel directory contents into the container at /app
COPY . /app/

# Make port 80 available to the world outside this container
EXPOSE 8000

# Define the command to run your FastAPI application
CMD ["uvicorn", "apimodel:app", "--host", "0.0.0.0", "--port", "8000"]