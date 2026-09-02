from mnist.domain.models import Vector, CachedGradient


def _numerical_derivative(network, x, y_true, perturb, restore, epsilon):
    perturb(epsilon)
    loss_plus = network.floss(network.forward(x), y_true)

    perturb(-epsilon)
    loss_minus = network.floss(network.forward(x), y_true)

    restore()
    return (loss_plus - loss_minus) / (2 * epsilon)


def numerical_gradient(network, x, y_true, epsilon=1e-5):
    """
    Calcula gradientes numéricamente para toda la red.

    Args:
        network: instancia de Network
        x: Vector de entrada
        y_true: Vector de salida esperada
        epsilon: perturbación infinitesimal

    Returns:
        dict {(layer_idx, neuron_idx): CachedGradient}, con la misma forma
        que el gradiente analítico cacheado en cada Perceptron, para poder
        compararlos campo a campo sin traducir claves.
    """
    gradients = {}

    for layer_idx, layer in enumerate(network.layers):
        for neuron_idx, neuron in enumerate(layer):

            original_weights = neuron.weights
            weight_grads = []
            for weight_idx in range(len(original_weights)):
                unit = Vector.one_hot(weight_idx, len(original_weights))

                def perturb(delta, unit=unit, original=original_weights):
                    neuron.weights = original + delta * unit

                def restore(original=original_weights):
                    neuron.weights = original

                weight_grads.append(
                    _numerical_derivative(network, x, y_true, perturb, restore, epsilon)
                )

            original_bias = neuron.bias

            def perturb_bias(delta, original=original_bias):
                neuron.bias = original + delta

            def restore_bias(original=original_bias):
                neuron.bias = original

            bias_grad = _numerical_derivative(network, x, y_true, perturb_bias, restore_bias, epsilon)

            gradients[(layer_idx, neuron_idx)] = CachedGradient(
                weights=Vector(weight_grads),
                bias=bias_grad
            )

    return gradients