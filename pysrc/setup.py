def configure_package():
	from numpy.distutils.misc_util import Configuration
	from numpy.distutils.misc_util import get_info

	config = Configuration('rsd')
	config.add_extension(
		'',
		['src/api.c', 'src/decoder.c', 'src/scheduler.c'],
		extra_info=get_info('npymath'),
		libraries=['avcodec', 'avformat', 'avutil']
	)

	return config

from numpy.distutils.core import setup

setup(configuration=configure_package)