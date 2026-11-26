import unittest
from math import inf, isclose, isnan

from neural_network.activations import identity, leaky_relu, relu, sigmoid, tanh
from neural_network.layers import Dense
from neural_network.numerics import _shape, dot, matvec


def add_vectors(a, b):
    return [x + y for x, y in zip(a, b)]


def scale_vector(alpha, a):
    return [alpha * x for x in a]


def combine(alpha, a, beta, b):
    return add_vectors(scale_vector(alpha, a), scale_vector(beta, b))


def add_matrices(a, b):
    return [add_vectors(row_a, row_b) for row_a, row_b in zip(a, b)]


def scale_matrix(alpha, a):
    return [scale_vector(alpha, row) for row in a]


class VectorTestCase(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected, places=7):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=places)


class TestDot(unittest.TestCase):
    def test_known_values(self):
        cases = [
            ([1, 2, 3], [4, 5, 6], 32),
            ([2, -1, 3], [1, 4, 2], 4),
            ([0, 0, 0], [3, 2, 1], 0),
            ([-1, -2], [-3, -4], 11),
            ([1, 0], [0, 1], 0),
            ([7], [3], 21),
            ([], [], 0),
            ([0.1, 0.2], [0.3, 0.4], 0.11),
        ]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(dot(a, b), expected)

    def test_is_commutative(self):
        a, b = [2, -5, 1.5], [4, 0.5, -2]
        self.assertEqual(dot(a, b), dot(b, a))

    def test_vector_with_itself_is_sum_of_squares(self):
        self.assertEqual(dot([3, 4], [3, 4]), 25)

    def test_unequal_lengths_raise(self):
        cases = [
            ([1, 2, 3], [1, 2]),
            ([1, 2], [1, 2, 3]),
            ([], [1]),
            ([1], []),
        ]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                with self.assertRaises(ValueError):
                    dot(a, b)

    def test_does_not_mutate_inputs(self):
        a, b = [1, 2], [3, 4]
        dot(a, b)
        self.assertEqual(a, [1, 2])
        self.assertEqual(b, [3, 4])


class TestShape(unittest.TestCase):
    def test_shapes(self):
        cases = [
            ([], (0, 0)),
            ([[1, 2], [3, 4]], (2, 2)),
            ([[1, 2, 3], [4, 5, 6]], (2, 3)),
            ([[1, 2], [3, 4], [5, 6]], (3, 2)),
            ([[], []], (2, 0)),
            ([[7]], (1, 1)),
        ]
        for matrix, expected in cases:
            with self.subTest(matrix=matrix):
                self.assertEqual(_shape(matrix), expected)

    def test_ragged_matrix_raises(self):
        with self.assertRaisesRegex(ValueError, "Ragged matrix: row 2 has 1 columns, expected 2"):
            _shape([[1, 2], [3, 4], [5]])


class TestMatvec(unittest.TestCase):
    def test_known_values(self):
        cases = [
            ([[1, 2, -1], [-2, 1, 3]], [2, 1, 3], [1, 6]),
            ([[1, 2], [3, 4]], [5, 6], [17, 39]),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [4, -2, 9], [4, -2, 9]),
            ([[0, 0], [0, 0], [0, 0]], [5, 7], [0, 0, 0]),
            ([[1, 2], [3, 4]], [0, 0], [0, 0]),
            ([[2], [-3], [5]], [4], [8, -12, 20]),
            ([], [1, 2], []),
            ([], [], []),
            ([[], [], []], [], [0.0, 0.0, 0.0]),
        ]
        for matrix, vector, expected in cases:
            with self.subTest(matrix=matrix, vector=vector):
                self.assertEqual(matvec(matrix, vector), expected)

    def test_single_row_matches_dot(self):
        row, vector = [2, -1, 3], [1, 4, 2]
        self.assertEqual(matvec([row], vector), [dot(row, vector)])

    def test_dimension_mismatches_raise(self):
        cases = [
            ([[1, 2, 3], [4, 5, 6]], [1, 2]),
            ([[1, 2], [3, 4]], [1, 2, 3]),
            ([[]], [1]),
            ([[1, 2]], []),
        ]
        for matrix, vector in cases:
            with self.subTest(matrix=matrix, vector=vector):
                with self.assertRaises(ValueError):
                    matvec(matrix, vector)

    def test_ragged_matrices_raise(self):
        cases = [
            ([[1, 2], [3]], [1, 1], "Ragged matrix: row 1 has 1 columns, expected 2"),
            ([[1, 2], [3, 4, 5]], [1, 1], "Ragged matrix: row 1 has 3 columns, expected 2"),
            ([[1], [2, 3]], [1], "Ragged matrix: row 1 has 2 columns, expected 1"),
            ([[], [1]], [], "Ragged matrix: row 1 has 1 columns, expected 0"),
            ([[1, 2], [3]], [1, 2, 3], "Ragged matrix: row 1"),
        ]
        for matrix, vector, message in cases:
            with self.subTest(matrix=matrix, vector=vector):
                with self.assertRaisesRegex(ValueError, message):
                    matvec(matrix, vector)

    def test_does_not_mutate_inputs(self):
        matrix, vector = [[1, 2], [3, 4]], [5, 6]
        matvec(matrix, vector)
        self.assertEqual(matrix, [[1, 2], [3, 4]])
        self.assertEqual(vector, [5, 6])


class TestFloatHandling(VectorTestCase):
    def test_dot_returns_float(self):
        cases = [
            ([1, 2, 3], [4, 5, 6], 32.0),
            ([], [], 0.0),
            ([1, 2.5, -3], [0.5, 4, 1.5], 6.0),
        ]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                result = dot(a, b)
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, expected)

    def test_dot_preserves_small_term_after_cancellation(self):
        self.assertEqual(dot([1e16, 1.0, -1e16], [1, 1, 1]), 1.0)

    def test_dot_accumulates_repeated_small_values_accurately(self):
        self.assertEqual(dot([0.1] * 10, [1] * 10), 1.0)

    def test_matvec_returns_floats(self):
        cases = [
            ([[1, 2], [3, 4]], [5, 6], [17.0, 39.0]),
            ([[1, 0.5], [2.5, -1]], [4, 2.0], [5.0, 8.0]),
        ]
        for matrix, vector, expected in cases:
            with self.subTest(matrix=matrix, vector=vector):
                result = matvec(matrix, vector)
                self.assertVectorAlmostEqual(result, expected)
                for entry in result:
                    self.assertIsInstance(entry, float)

    def test_matvec_preserves_small_term_after_cancellation(self):
        result = matvec([[1e16, 1.0, -1e16], [1.0, 1.0, 1.0]], [1, 1, 1])
        self.assertEqual(result, [1.0, 3.0])


class TestDotBilinearity(VectorTestCase):
    def test_linear_in_first_argument(self):
        x, y, z = [1, -2, 0.5], [3, 4, -1], [2, 0.25, 5]
        for alpha, beta in [(2, 3), (-1.5, 0.5), (0, 4), (3, 0), (0, 0)]:
            with self.subTest(alpha=alpha, beta=beta):
                self.assertAlmostEqual(
                    dot(combine(alpha, x, beta, y), z),
                    alpha * dot(x, z) + beta * dot(y, z),
                )

    def test_linear_in_second_argument(self):
        x, y, z = [2, 0.25, 5], [1, -2, 0.5], [3, 4, -1]
        for alpha, beta in [(2, 3), (-1.5, 0.5), (0, 4), (3, 0), (0, 0)]:
            with self.subTest(alpha=alpha, beta=beta):
                self.assertAlmostEqual(
                    dot(x, combine(alpha, y, beta, z)),
                    alpha * dot(x, y) + beta * dot(x, z),
                )

    def test_additive_in_each_argument(self):
        x, y, z = [1, 2, 3], [-4, 5, 0.5], [0.5, -1, 2]
        self.assertAlmostEqual(dot(add_vectors(x, y), z), dot(x, z) + dot(y, z))
        self.assertAlmostEqual(dot(x, add_vectors(y, z)), dot(x, y) + dot(x, z))

    def test_homogeneous_in_each_argument(self):
        x, y = [1.5, -2, 3], [4, 0.5, -1]
        for alpha in [0, 1, -1, 2.5, -0.75]:
            with self.subTest(alpha=alpha):
                self.assertAlmostEqual(dot(scale_vector(alpha, x), y), alpha * dot(x, y))
                self.assertAlmostEqual(dot(x, scale_vector(alpha, y)), alpha * dot(x, y))

    def test_scaling_both_arguments_scales_by_product(self):
        x, y = [1, -3, 2], [0.5, 4, -2]
        self.assertAlmostEqual(
            dot(scale_vector(2, x), scale_vector(-3, y)), -6 * dot(x, y)
        )


class TestMatvecLinearity(VectorTestCase):
    def test_homogeneity(self):
        w = [[1, 2, -1], [-2, 1, 3]]
        x = [2, -1, 0.5]
        for alpha in [0, 1, -1, 2, -3.5, 0.25]:
            with self.subTest(alpha=alpha):
                self.assertVectorAlmostEqual(
                    matvec(w, scale_vector(alpha, x)),
                    scale_vector(alpha, matvec(w, x)),
                )

    def test_linear_combination(self):
        w = [[1, 2, -1], [-2, 1, 3], [0.5, -0.25, 4]]
        x, y = [2, 1, 3], [-1, 4, 0.5]
        for alpha, beta in [(2, 3), (-1.5, 0.5), (0, 4), (3, 0), (0, 0), (1, -1), (1, 1)]:
            with self.subTest(alpha=alpha, beta=beta):
                self.assertVectorAlmostEqual(
                    matvec(w, combine(alpha, x, beta, y)),
                    combine(alpha, matvec(w, x), beta, matvec(w, y)),
                )

    def test_linear_combination_for_various_shapes(self):
        cases = [
            ([[3]], [2], [-5]),
            ([[1, 2, -1]], [2, 1, 3], [0, -4, 1]),
            ([[2], [-3], [5], [0.5]], [4], [-2]),
            ([[1, 2], [3, 4], [5, 6]], [1, -1], [2, 3]),
            ([[0, 0], [0, 0]], [1, 2], [3, 4]),
            ([[0.5, -1.25], [2.75, 0.1]], [0.3, -2], [4, 0.75]),
        ]
        for w, x, y in cases:
            with self.subTest(w=w, x=x, y=y):
                self.assertVectorAlmostEqual(
                    matvec(w, combine(2.5, x, -1.5, y)),
                    combine(2.5, matvec(w, x), -1.5, matvec(w, y)),
                )

    def test_zero_combination_gives_zero_vector(self):
        w = [[1, 2], [3, 4], [5, 6]]
        result = matvec(w, combine(0, [1, 2], 0, [3, 4]))
        self.assertVectorAlmostEqual(result, [0, 0, 0])

    def test_linear_combination_of_three_vectors(self):
        w = [[1, 2, -1], [-2, 1, 3]]
        x, y, z = [1, 0, 2], [3, -1, 1], [0.5, 2, -4]
        combined = add_vectors(combine(2, x, -1, y), scale_vector(0.5, z))
        expected = add_vectors(
            combine(2, matvec(w, x), -1, matvec(w, y)),
            scale_vector(0.5, matvec(w, z)),
        )
        self.assertVectorAlmostEqual(matvec(w, combined), expected)

    def test_linear_in_matrix_argument(self):
        a = [[1, 2], [3, -1]]
        b = [[0.5, -2], [4, 0.25]]
        x = [2, -3]
        for alpha, beta in [(2, 3), (-1, 0.5), (0, 0)]:
            with self.subTest(alpha=alpha, beta=beta):
                self.assertVectorAlmostEqual(
                    matvec(add_matrices(scale_matrix(alpha, a), scale_matrix(beta, b)), x),
                    combine(alpha, matvec(a, x), beta, matvec(b, x)),
                )


class TestDense(VectorTestCase):
    def test_forward_known_values(self):
        cases = [
            ([[1, 2], [3, 4]], [0.5, -1], [5, 6], [17.5, 38.0]),
            ([[1, 2], [3, 4]], [-20, 1], [5, 6], [0.0, 40.0]),
            ([[1, 0], [0, 1]], [0, 0], [-2, 3], [0.0, 3.0]),
            ([[0, 0], [0, 0]], [-1, 2], [9, -9], [0.0, 2.0]),
            ([[2, -1, 3]], [0.5], [1, 4, 2], [4.5]),
            ([[], []], [-1, 2], [], [0.0, 2.0]),
            ([], [], [], []),
        ]
        for weights, biases, inputs, expected in cases:
            with self.subTest(weights=weights, biases=biases, inputs=inputs):
                layer = Dense(weights, biases, relu)
                self.assertVectorAlmostEqual(layer.forward(inputs), expected)

    def test_empty_layer_rejects_input_that_does_not_match_input_size(self):
        with self.assertRaises(ValueError):
            Dense([], [], relu).forward([1, 2])

    def test_forward_returns_floats(self):
        layer = Dense([[0.5, 0.25], [1.5, -0.1]], [0.1, -3], relu)
        result = layer.forward([2, 4])
        self.assertVectorAlmostEqual(result, [2.1, 0.0])
        for entry in result:
            self.assertIsInstance(entry, float)

    def test_invalid_construction_raises(self):
        cases = [
            ([[1, 2], [3]], [0, 0], "Ragged matrix: row 1"),
            ([[1, 2], [3, 4]], [0], None),
            ([[1, 2]], [0, 0], None),
            ([], [0], None),
        ]
        for weights, biases, message in cases:
            with self.subTest(weights=weights, biases=biases):
                if message is None:
                    with self.assertRaises(ValueError):
                        Dense(weights, biases, relu)
                else:
                    with self.assertRaisesRegex(ValueError, message):
                        Dense(weights, biases, relu)

    def test_forward_input_mismatches_raise(self):
        layer = Dense([[1, 2], [3, 4]], [0, 0], relu)
        for inputs in ([1, 2, 3], []):
            with self.subTest(inputs=inputs):
                with self.assertRaises(ValueError):
                    layer.forward(inputs)

    def test_forward_does_not_mutate_inputs(self):
        layer = Dense([[1, 2], [3, 4]], [0.5, -1], relu)
        inputs = [5, 6]
        layer.forward(inputs)
        self.assertEqual(inputs, [5, 6])

    def test_repeated_forward_calls_give_same_result(self):
        layer = Dense([[1, -2], [0.5, 3]], [1, -1], relu)
        first = layer.forward([4, 1])
        second = layer.forward([4, 1])
        self.assertVectorAlmostEqual(first, second)

    def test_mutating_original_weights_does_not_affect_layer(self):
        weights = [[1, 2], [3, 4]]
        layer = Dense(weights, [0.5, -1], relu)
        weights.append([100, 100])
        self.assertVectorAlmostEqual(layer.forward([5, 6]), [17.5, 38.0])
        weights[0] = [100, 100]
        self.assertVectorAlmostEqual(layer.forward([5, 6]), [17.5, 38.0])
        weights[1][0] = 100
        self.assertVectorAlmostEqual(layer.forward([5, 6]), [17.5, 38.0])

    def test_mutating_original_biases_does_not_affect_layer(self):
        biases = [0.5, -1]
        layer = Dense([[1, 2], [3, 4]], biases, relu)
        biases[0] = 100
        self.assertVectorAlmostEqual(layer.forward([5, 6]), [17.5, 38.0])
        biases.append(100)
        self.assertVectorAlmostEqual(layer.forward([5, 6]), [17.5, 38.0])

class TestConfigurableActivations(VectorTestCase):
    def test_library_activations_are_applied(self):
        weights, biases, inputs = [[1, 2], [3, 4]], [-20, 1], [5, 6]
        cases = [
            ("identity", identity, [-3.0, 40.0]),
            ("relu", relu, [0.0, 40.0]),
            ("leaky_relu", leaky_relu, [-0.03, 40.0]),
            ("leaky_relu_alpha_0.1", lambda x: leaky_relu(x, 0.1), [-0.3, 40.0]),
            ("sigmoid", sigmoid, [0.04742587317756678, 1.0]),
            ("tanh", tanh, [-0.9950547536867305, 1.0]),
        ]
        for name, activation, expected in cases:
            with self.subTest(activation=name):
                layer = Dense(weights, biases, activation)
                self.assertVectorAlmostEqual(layer.forward(inputs), expected)

    def test_activation_is_applied_after_bias_addition(self):
        layer = Dense([[1, 2], [3, 4]], [0.5, -1], lambda x: 2 * x)
        self.assertVectorAlmostEqual(layer.forward([5, 6]), [35.0, 76.0])

    def test_activation_receives_each_preactivation_in_order(self):
        seen = []

        def recording_activation(x):
            seen.append(x)
            return x + 100

        layer = Dense([[1, 2], [3, 4]], [-20, 1], recording_activation)
        result = layer.forward([5, 6])
        self.assertEqual(seen, [-3.0, 40.0])
        self.assertVectorAlmostEqual(result, [97.0, 140.0])

    def test_callable_object_activation(self):
        class Scale:
            def __init__(self, factor):
                self.factor = factor

            def __call__(self, x):
                return self.factor * x

        layer = Dense([[1, 2], [3, 4]], [0.5, -1], Scale(0.5))
        self.assertVectorAlmostEqual(layer.forward([5, 6]), [8.75, 19.0])

    def test_layer_uses_the_activation_it_was_given(self):
        def custom(x):
            return x

        layer = Dense([[1]], [0], custom)
        self.assertIs(layer.activation_function, custom)

    def test_non_callable_activation_raises_type_error(self):
        with self.assertRaises(TypeError):
            Dense([[1, 2], [3, 4]], [0, 0], None).forward([5, 6])


class TestActivations(VectorTestCase):
    def test_relu_values(self):
        cases = [
            (3.5, 3.5),
            (-2.5, 0.0),
            (0, 0.0),
            (0.0, 0.0),
            (-0.0, 0.0),
            (7, 7),
            (-7, 0.0),
            (1e-12, 1e-12),
            (-1e16, 0.0),
        ]
        for x, expected in cases:
            with self.subTest(x=x):
                self.assertEqual(relu(x), expected)

    def test_identity_values(self):
        cases = [
            (3.5, 3.5),
            (-2.5, -2.5),
            (0, 0.0),
            (0.0, 0.0),
            (-0.0, 0.0),
            (7, 7),
            (1e16, 1e16),
            (-1e16, -1e16),
        ]
        for x, expected in cases:
            with self.subTest(x=x):
                self.assertEqual(identity(x), expected)

    def test_sigmoid_values(self):
        cases = [
            (0, 0.5),
            (1, 0.7310585786300049),
            (-1, 0.2689414213699951),
            (2, 0.8807970779778823),
            (-2, 0.11920292202211755),
            (0.5, 0.6224593312018546),
            (100, 1.0),
        ]
        for x, expected in cases:
            with self.subTest(x=x):
                self.assertAlmostEqual(sigmoid(x), expected)

    def test_sigmoid_small_values_are_nonzero(self):
        cases = [
            (-100, 3.720075976020836e-44),
            (-710, 4.47628622567513e-309),
        ]
        for x, expected in cases:
            with self.subTest(x=x):
                result = sigmoid(x)
                self.assertNotEqual(result, 0.0)
                self.assertTrue(isclose(result, expected, rel_tol=1e-9, abs_tol=0.0))

    def test_sigmoid_underflows_to_zero(self):
        self.assertEqual(sigmoid(-1000), 0.0)

    def test_sigmoid_properties(self):
        xs = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
        outputs = [sigmoid(x) for x in xs]
        for x, output in zip(xs, outputs):
            with self.subTest(x=x):
                self.assertGreater(output, 0)
                self.assertLess(output, 1)
                self.assertAlmostEqual(sigmoid(-x), 1 - output)
        for smaller, larger in zip(outputs, outputs[1:]):
            self.assertLess(smaller, larger)

    def test_tanh_values(self):
        cases = [
            (0, 0.0),
            (1, 0.7615941559557649),
            (-1, -0.7615941559557649),
            (0.5, 0.46211715726000974),
            (-0.5, -0.46211715726000974),
            (2, 0.9640275800758169),
            (100, 1.0),
            (-100, -1.0),
        ]
        for x, expected in cases:
            with self.subTest(x=x):
                self.assertAlmostEqual(tanh(x), expected)

    def test_tanh_properties(self):
        xs = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
        outputs = [tanh(x) for x in xs]
        for x, output in zip(xs, outputs):
            with self.subTest(x=x):
                self.assertGreater(output, -1)
                self.assertLess(output, 1)
                self.assertAlmostEqual(tanh(-x), -output)
        for smaller, larger in zip(outputs, outputs[1:]):
            self.assertLess(smaller, larger)

    def test_leaky_relu_values(self):
        default_cases = [
            (3.5, 3.5),
            (-2.5, -0.025),
            (0, 0.0),
            (-100, -1.0),
            (7, 7),
        ]
        for x, expected in default_cases:
            with self.subTest(x=x, alpha="default"):
                self.assertAlmostEqual(leaky_relu(x), expected)
        custom_cases = [
            (-2.5, 0.1, -0.25),
            (-2.5, 0.5, -1.25),
            (-2.5, 0, 0.0),
            (-2.5, 1, -2.5),
            (3.5, 0.1, 3.5),
            (3.5, 0.5, 3.5),
        ]
        for x, alpha, expected in custom_cases:
            with self.subTest(x=x, alpha=alpha):
                self.assertAlmostEqual(leaky_relu(x, alpha), expected)

class TestExtremeInputs(VectorTestCase):
    def test_bounded_activations_saturate_for_large_inputs(self):
        cases = [
            (sigmoid, 100, 1.0),
            (sigmoid, 1000, 1.0),
            (sigmoid, 1e308, 1.0),
            (sigmoid, inf, 1.0),
            (sigmoid, -1000, 0.0),
            (sigmoid, -1e308, 0.0),
            (sigmoid, -inf, 0.0),
            (tanh, 100, 1.0),
            (tanh, 1000, 1.0),
            (tanh, 1e308, 1.0),
            (tanh, inf, 1.0),
            (tanh, -100, -1.0),
            (tanh, -1000, -1.0),
            (tanh, -1e308, -1.0),
            (tanh, -inf, -1.0),
        ]
        for function, x, expected in cases:
            with self.subTest(function=function.__name__, x=x):
                self.assertEqual(function(x), expected)

    def test_bounded_activations_propagate_nan(self):
        for function in (sigmoid, tanh):
            with self.subTest(function=function.__name__):
                self.assertTrue(isnan(function(float("nan"))))

    def test_unbounded_activations_handle_extreme_inputs(self):
        cases = [
            (relu, 1e308, 1e308),
            (relu, -1e308, 0.0),
            (relu, inf, inf),
            (relu, -inf, 0.0),
            (identity, 1e308, 1e308),
            (identity, -1e308, -1e308),
            (identity, inf, inf),
            (identity, -inf, -inf),
            (leaky_relu, 1e308, 1e308),
            (leaky_relu, -1000, -10.0),
            (leaky_relu, inf, inf),
            (leaky_relu, -inf, -inf),
        ]
        for function, x, expected in cases:
            with self.subTest(function=function.__name__, x=x):
                self.assertEqual(function(x), expected)

    def test_dense_with_bounded_activation_saturates_instead_of_overflowing(self):
        cases = [
            (sigmoid, [0.0], [1.0]),
            (tanh, [-1.0], [1.0]),
        ]
        for activation, negative, positive in cases:
            with self.subTest(activation=activation.__name__):
                layer = Dense([[1000]], [0], activation)
                self.assertVectorAlmostEqual(layer.forward([-1]), negative)
                self.assertVectorAlmostEqual(layer.forward([1]), positive)

    def test_dot_with_extreme_cancellation(self):
        self.assertEqual(dot([1e308, -1e308], [1, 1]), 0.0)

    def test_dot_underflows_to_zero(self):
        self.assertEqual(dot([1e-200], [1e-200]), 0.0)


class TestDenseInterface(unittest.TestCase):
    def test_layer_interface(self):
        cases = [
            ([[1, 2, 3], [4, 5, 6]], [0, 0], 3, 2, 8),
            ([[1, 2], [3, 4]], [0, 0], 2, 2, 6),
            ([[2, -1, 3]], [0.5], 3, 1, 4),
            ([[2], [-3], [5]], [0, 0, 0], 1, 3, 6),
            ([[1, 2], [3, 4], [5, 6]], [0, 0, 0], 2, 3, 9),
            ([], [], 0, 0, 0),
            ([[], []], [-1, 2], 0, 2, 2),
        ]
        for weights, biases, input_size, output_size, parameter_count in cases:
            with self.subTest(weights=weights):
                layer = Dense(weights, biases, relu)
                self.assertIsInstance(layer.input_size, int)
                self.assertIsInstance(layer.output_size, int)
                self.assertEqual(layer.input_size, input_size)
                self.assertEqual(layer.output_size, output_size)
                self.assertEqual(layer.parameter_count(), parameter_count)


if __name__ == "__main__":
    unittest.main()
