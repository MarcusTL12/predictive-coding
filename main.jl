using LinearAlgebra
using IntelVectorMath

struct NetworkParams
    n_layers::Int
    total_vector_length::Int
    total_weight_length::Int
    layer_sizes::Vector{Int}
    vector_offsets::Vector{Int}
    weight_offsets::Vector{Int}
end

function NetworkParams(layer_sizes)
    n_layers = length(layer_sizes)

    vector_offsets = zeros(Int, n_layers)
    weight_offsets = zeros(Int, n_layers)

    for i in 1:(n_layers-1)
        vector_offsets[i+1] = vector_offsets[i] + layer_sizes[i]
        weight_offsets[i+1] = weight_offsets[i] + layer_sizes[i] * layer_sizes[i+1]
    end

    total_vector_length = vector_offsets[end] + layer_sizes[end]
    total_weight_length = pop!(weight_offsets)

    NetworkParams(n_layers, total_vector_length, total_weight_length,
        layer_sizes, vector_offsets, weight_offsets)
end

function get_vectors_layer(p::NetworkParams, x, l)
    start = p.vector_offsets[l]
    stop = start + p.layer_sizes[l] - 1

    @view x[:, (begin+start):(begin+stop)]
end

function get_weights_layer(p::NetworkParams, w, l)
    start = p.weight_offsets[l]
    stop = start + p.layer_sizes[l] * p.layer_sizes[l+1] - 1

    reshape(w[(begin+start):(begin+stop)], p.layer_sizes[l+1], p.layer_sizes[l])
end

# σ(x) = 1 / (1 + exp(-4x))
function σ!(p::NetworkParams, z, x, b)
    n_inputs = size(z, 1)

    @assert size(x, 1) == n_inputs

    @assert size(z, 2) == p.total_vector_length
    @assert size(x, 2) == p.total_vector_length
    @assert length(b) == p.total_vector_length

    # @inbounds for i in 1:p.total_vector_length
    #     z[i] = 1 / (1 + exp(-4 * (x[i] + b[i])))
    # end

    # z = -4 (x + b)
    @inbounds for j in 1:p.total_vector_length, i in 1:n_inputs
        z[i, j] = -4 * (x[i, j] + b[j])
    end

    # z = exp(z)
    IVM.exp!(z)

    # z = 1 / (1 + z)
    @inbounds for i in eachindex(z)
        z[i] = 1 / (1 + z[i])
    end
end

function compute_predictions_type1!(p::NetworkParams, μ, z, w, x, b)
    n_inputs = size(μ, 1)

    @assert size(z, 1) == n_inputs
    @assert size(x, 1) == n_inputs

    @assert size(μ, 2) == p.total_vector_length
    @assert size(z, 2) == p.total_vector_length
    @assert size(x, 2) == p.total_vector_length
    @assert length(b) == p.total_vector_length

    σ!(p, z, x, b)

    for l in 1:(p.n_layers-1)
        w_l = get_weights_layer(p, w, l)
        z_l = get_vectors_layer(p, z, l)
        μ_lp1 = get_vectors_layer(p, μ, l + 1)

        mul!(μ_lp1, z_l, w_l')
    end
end
