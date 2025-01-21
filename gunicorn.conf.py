import multiprocessing
import os

# Get the directory containing this file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Binding
bind = "127.0.0.1:5000"

# Workers
workers = multiprocessing.cpu_count() * 2 + 1

# Logging
accesslog = os.path.join(base_dir, "logs/access.log")
errorlog = os.path.join(base_dir, "logs/error.log")
loglevel = "info"

# Timeout
timeout = 60

# SSL (if needed)
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile"

# Worker Options
worker_class = "sync"