def untuple(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def intervention(intervention_layer, intervene_at, patching_vector):
    def edit_output(layer, output):
        if layer != intervention_layer:
            return output
        untuple(output)[intervene_at] = patching_vector
        return output

    return edit_output
