from distutils.core import setup

setup(
    name='resnet-tensorflow',
    version='0.0.1',
    packages=['resnet'],
    url='https://github.com/AndersonJo/residual-network',
    license='Apache 2.0',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    install_requires=['tensorflow', 'hdfs', 'keras', 'numpy', 'scipy'],
    python_requires='>=3',
    author='Chang Min Jo (Anderson Jo)',
    author_email='a141890@gmail.com',
    description='Deep Residual Neural Network',
    keywords=['tensorflow', 'resnet', 'residual', 'neural network', 'cifar', 'cifar-10']
)
