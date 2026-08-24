using LinearAlgebra

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

function get_vector_layer(p::NetworkParams, x, l)
    start = p.vector_offsets[l]
    stop = start + p.layer_sizes[l] - 1

    @view x[(begin+start):(begin+stop)]
end

function get_weight_layer(p::NetworkParams, w, l)
    start = p.weight_offsets[l]
    stop = start + p.layer_sizes[l] * p.layer_sizes[l + 1] - 1

    reshape(w[(begin+start):(begin+stop)], p.layer_sizes[l + 1], p.layer_sizes[l])
end

function compute_errors!(params::NetworkParams, e, x, μ)
    @. e = x - μ
end

function compute_energy(params::NetworkParams, x)

end
