

# First version of values
def Valora(df, target_column='ROI',normaliza=True,
           categorias=['bajo_moderado', 'bajo', 'bajo_alto','medio_moderado', 'medio', 'medio_alto',
                       'alto_moderado', 'alto', 'muy_alto']):
	"""
		Function which values a column, in this case the use-case used is with the ROI( Return of Investment ).
		It will be created three differents arrays one for the values, another one for the category and the last one
		for the anwser ( Yest or No) .
		Categories by default ( Spanish ):
		[YES]: muy alto / alto / alto moderado / medio alto / medio / medio moderado
		[NO]: bajo alto / bajo / bajo moderado / deficiente ( Negative value ) / neutro ( When it's 0 )
		:param df: Dataframe pandas
		:return: Dataframe of pandas with the new valued-columns
	"""

	# Extract the ROI
	ROI = df[target_column]

	# check that it's an integer or a float

	# obtain the positive values
	positivos = [i for i in ROI if i > 0]

	# extract negatives
	negativos_neutros = [i for i in ROI if i <= 0]
	negativos_neutros_discretizado = []
	negativos_neutros_categoria = []
	negativos_neutros_respuesta = []
	# build the array of categories
	for i in negativos_neutros:
		if i < 0:
			negativos_neutros_discretizado.append(-2)
			negativos_neutros_categoria.append('deficiente')
		else:
			negativos_neutros_discretizado.append(-1)
			negativos_neutros_categoria.append('Neutro')
		negativos_neutros_respuesta.append("NO")

	# Divide in 10 categories and extract the cut points
	data_dis = {'bins':6, 'codificacion':"one-hot"}
	# _, pos_dis, enc = Bin_dis.kbins_algo(positivos, nbins=10, modo="array")
	_, pos_dis, enc = Bin_dis.kbins_algo(positivos, data_dis)
	cut_points = enc.bin_edges[0]

	# Creamos el array de categorias
	positivos_categorias = [categorias[i[0].indices[0]] for i in pos_dis]

	# Establecemos la respuesta de la pregunta de si sube el roi
	positivos_respuesta = []
	for i in positivos_discretizados:
		if i[0].indices >= 3:
			positivos_respuesta.append("SI")
		else:
			positivos_respuesta.append("NO")

	# proceso de union
	total_discretizados = []
	total_categorias = []
	total_respuesta = []

	# Revertirmos los arrays para hacer pop
	positivos_categorias.reverse()
	positivos_respuesta.reverse()
	positivos_discretizados_lista = positivos_discretizados.indices.tolist()
	positivos_discretizados_lista.reverse()
	negativos_neutros_categoria.reverse()
	negativos_neutros_respuesta.reverse()
	negativos_neutros_discretizado.reverse()

	for i in ROI:
		# Insertamos en variables temporales
		if i > 0:
			# tenemos que hacer el pop a mano , dado que en numpy no existe
			discretizado = positivos_discretizados_lista.pop()
			categorias = positivos_categorias.pop()
			respuesta = positivos_respuesta.pop()
		else:
			discretizado = negativos_neutros_discretizado.pop()
			categorias = negativos_neutros_categoria.pop()
			respuesta = negativos_neutros_respuesta.pop()

		# guardamos en los arrays de destino
		total_discretizados.append(discretizado)
		total_categorias.append(categorias)
		total_respuesta.append(respuesta)


	# Annadimos las nuevas columnas al df
	df[columna_target+'_discretizada_valoracion'] = total_discretizados
	df[columna_target+'_discretizada_valoracion_categoria'] = total_categorias
	df[columna_target+'_discretizada_valoracion_respuesta'] = total_respuesta

	# normalizamos si se solicita
	if normaliza:
		df[columna_target+'_discretizada_valoracion'] += 2

	return df
