def label_trace(*args):
	raise DeprecationWarning("""`label_trace` is no longer supported. Instead use:
			>>> from infostop import Infostop
			>>> model = Infostop()
			>>> labels = model.fit_predict(my_data)""")
	
def label_static_points(*args):
	raise DeprecationWarning("""`label_static_points` is no longer supported. Instead use:
			>>> from infostop import SpatialInfomap
			>>> model = SpatialInfomap()
			>>> labels = model.fit_predict(my_data)""")

def label_network(*args):
	raise DeprecationWarning("""`label_network` is no longer supported. Instead use:
			>>> from infostop import Infostop
			>>> model = Infostop()
			>>> labels = model.fit_predict(my_data)""")