from setuptools import setup, find_packages

setup(
    name='vector-storage-app',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for exploring different vector storage and embedding techniques.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scikit-learn',
        'faiss-cpu',  # or 'faiss-gpu' if you are using GPU
        'torch',  # if using PyTorch for embeddings
        'transformers'  # if using Hugging Face Transformers
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)