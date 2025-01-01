# Use an official Python runtime as a parent image
FROM python:3.12.3

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose port 5500 to the outside world
EXPOSE 5500

# Run the Python HTTP server when the container launches
CMD ["python", "-m", "http.server", "5500"]