from mnist.domain.models import Vector


def numerical_gradient(network, x, y_true, epsilon=1e-5):
    """
    Calcula gradientes numéricamente para toda la red.
    
    Args:
        network: instancia de Network
        x: Vector de entrada
        y_true: Vector de salida esperada
        epsilon: perturbación infinitesimal
    
    Returns:
        dict con estructura: {(layer_idx, neuron_idx, weight_idx): grad_numerico}
        además {(layer_idx, neuron_idx, 'bias'): grad_numerico}
    """
    gradients = {}
    
    # Itera sobre todas las capas
    for layer_idx, layer in enumerate(network.layers):
        
        # Itera sobre todas las neuronas en la capa
        for neuron_idx, neuron in enumerate(layer):

            # Gradientes de pesos
            original_weights = neuron.weights
            for weight_idx in range(len(original_weights)):
                unit = Vector.one_hot(weight_idx, len(original_weights))

                # Perturbo: w + ε
                neuron.weights = original_weights + epsilon * unit
                output_plus = network.forward(x)
                loss_plus = network.floss(output_plus, y_true)

                # Perturbo: w - ε
                neuron.weights = original_weights - epsilon * unit
                output_minus = network.forward(x)
                loss_minus = network.floss(output_minus, y_true)

                # Restauro
                neuron.weights = original_weights

                # Gradiente numérico
                grad_numerical = (loss_plus - loss_minus) / (2 * epsilon)
                gradients[(layer_idx, neuron_idx, weight_idx)] = grad_numerical

            # Gradiente de bias
            original_bias = neuron.bias

            # Perturbo: b + ε
            neuron.bias = original_bias + epsilon
            output_plus = network.forward(x)
            loss_plus = network.floss(output_plus, y_true)

            # Perturbo: b - ε
            neuron.bias = original_bias - epsilon
            output_minus = network.forward(x)
            loss_minus = network.floss(output_minus, y_true)
            
            # Restauro
            neuron.bias = original_bias
            
            # Gradiente numérico
            grad_numerical = (loss_plus - loss_minus) / (2 * epsilon)
            gradients[(layer_idx, neuron_idx, 'bias')] = grad_numerical
    
    return gradients