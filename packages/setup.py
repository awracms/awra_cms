from setuptools import setup,find_packages

setup(name='awrams',
      packages=['awrams', 'awrams.benchmarking', 'awrams.calibration',
                'awrams.cluster','awrams.models', 'awrams.simulation',
                'awrams.utils','awrams.visualisation'],
      version='1.2',
      description='AWRA Modelling System (Community Release), version 1.2',
      url='https://github.com/awracms/awra_cms',
      author='awrams team',
      author_email='awrams@bom.gov.au',
      license='MIT',
      zip_safe=False,
      include_package_data=True,
      setup_requires=['nose>=1.3.3'],
      test_suite='nose.collector',
      tests_require=['nose'],
      )
