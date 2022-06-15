using Flux
using Gadfly
using ProgressBars
using Cairo
using DataFrames
using Statistics
using CUDA
using DelimitedFiles

include("parser.jl")

CUDA.allowscalar(false)
const data_path = "/u/sauves/leucegene-shared/Data"
const output_path = "output"

theme_gadfly = Theme(
    minor_label_font="Arial", 
    major_label_font="Arial", 
    key_title_font="Arial",
    key_label_font="Arial",
    point_label_font="Arial",
    minor_label_color="black", 
    major_label_color="black",
    key_title_color="black",
    key_label_color="black",
    point_label_color="black",
    minor_label_font_size=8pt,
    background_color="white"
)
Gadfly.push_theme(theme_gadfly)

function load_data()
    return @time parser("$(data_path)/SIGNATURES/LSC17_lgn_pronostic_expressions.csv")
end

function prep_data(data::Parseddata; shuffle=true, device=gpu)
    m = length(data.CL_Name)
    n = length(data.gene_symbol)

    outs = Array{Float32,2}(undef, (1, n * m))
    genes = Array{Int32,1}(undef, n * m)
    cell_lines = Array{Int32,1}(undef, n * m)
    small_molecules = Array{Int32,1}(undef, n * m)

    gi = 0
    ci = 0
    mi = 0
    unique_genes = Dict{String,Int32}()
    unique_cell_lines = Dict{String,Int32}()
    unique_small_molecules = Dict{String,Int32}()
    for i in ProgressBar(1:m)
        for j in 1:n
            index = (i - 1) * n + j
            
            outs[1, index] = data.data_log[j, i]

            genes[index] = get!(unique_genes, data.gene_symbol[j]) do 
                gi += 1
            end

            cell_lines[index] = get!(unique_cell_lines, data.CL_Name[i]) do 
                ci += 1
            end

            small_molecules[index] = get!(unique_small_molecules, data.SM_Name[i]) do 
                mi += 1
            end
        end
    end

    if shuffle
        genes, cell_lines, small_molecules, outs = shuffle_data(genes, cell_lines, small_molecules, outs)
    end

    return (device(genes), device(cell_lines), device(small_molecules)), device(outs)
end

function shuffle_data(genes, cell_lines, small_molecules, outs)
    index = rand(1:length(genes), length(genes))

    genes2 = genes[index]
    cell_lines2 = cell_lines[index]
    small_molecules2 = small_molecules[index]
    outs2 = reshape(outs[index], (1, length(outs)))

    return genes2, cell_lines2, small_molecules2, outs2
end

### Plots ###

function plot_loss(folder, loss_data)
    l = length(loss_data)
    draw(
        PNG("$(output_path)/$(folder)/loss_curve.png"), 
        plot(
            x=1:l, y=loss_data, Geom.line, 
            Guide.title("Loss per epoch (loss at last 10 epochs = $(mean(loss_data[(l - 9):l])))"), 
            Guide.xlabel("Epoch"), Guide.ylabel("Loss")
        )
    )
end

function get_obtained_expected(X, Y, network, device)
    nb = 10000
    index = rand(1:length(Y), nb)

    obtained = cpu(Y)[index]

    X_cpu = cpu(X)
    genes = X_cpu[1][index]
    cells = X_cpu[2][index]
    molecules = X_cpu[3][index]
    expected = cpu(network((device(genes), device(cells), device(molecules))))

    return obtained, expected[1,:]
end

function plot_accuracy(folder, epoch, obtained, expected)
    corP = Statistics.cor(obtained, expected)
    draw(
        PNG("$(output_path)/$(folder)/accuracy_at_epoch_$epoch.png"), 
        plot(
            x=expected, y=obtained, Guide.title("Accuracy at epoch $epoch (r = $corP)"), 
            Guide.xlabel("Expected"), Guide.ylabel("Obtained"), Geom.abline, 
            Geom.hexbin(xbincount=100, ybincount=100)
        )
    )
end

function plot_embedding(embedding, name, folder, labels)
    draw(
        PNG("$(output_path)/$(folder)/$name.png"), 
        plot(
            x=cpu(embedding.weight)[1,:], y=cpu(embedding.weight)[2,:], label=labels,
            Guide.title("Embedding ($name)"), Guide.xlabel("x"), Guide.ylabel("y"), 
            Geom.point, Geom.label
        )
    )

    touch("$(output_path)/$(folder)/$name.txt")
    writedlm("$(output_path)/$(folder)/$name.txt", cpu(embedding.weight))
end

function embeddings_dims(data_matrix, corP, iterator_dims, folder)
    X, Y = prep_data(data_matrix)
    data = Flux.Data.DataLoader((X, Y), batchsize=524288)
    
    for i in ProgressBar(iterator_dims)
        corP[i - 1] = train_plot(data, X, Y, (i, i, i), "$folder/d$i", 50, data_matrix)
    end

    draw(
        PNG("$(output_path)/$(folder)/accuracy_at_dim.png"), 
        plot(
            x=collect(iterator_dims), y=corP, Guide.title("Accuracy at dimension"), 
            Guide.xlabel("Dimension of embeddings"), Guide.ylabel("r^2"),
            Geom.point, Geom.smooth(method=:loess, smoothing=0.9)
        )
    )
end

function embeddings_dims2(data_matrix, corP, iterator_dims, folder)
    X, Y = prep_data(data_matrix)
    data = Flux.Data.DataLoader((X, Y), batchsize=524288)
    
    for i in ProgressBar(iterator_dims)
        corP[i - 1] = train_plot(data, X, Y, (i, 50, 50), "$folder/d$i", data_matrix, 50)
    end

    draw(
        PNG("$(output_path)$(folder)/accuracy_at_dim.png"), 
        plot(
            x=collect(iterator_dims), y=corP, Guide.title("Accuracy at dimension"), 
            Guide.xlabel("Dimension of embeddings"), Guide.ylabel("r^2"),
            Geom.point, Geom.smooth(method=:loess, smoothing=0.9)
        )
    )
end

#########

function index_file(folder, embeddings_dims, batchsize, training_rate, neural_network)
    open("$(output_path)/$(folder)/index.txt", "w") do io
        print(io, 
            "Nombre de dimensions par embedding :\n" * 
            "- Gene layer : $(embeddings_dims[1])\n" * 
            "- Cell line layer : $(embeddings_dims[2])\n" *
            "- Molecule layer : $(embeddings_dims[3])\n\n" *
            "Dimensions du reseau de neurones :\n$neural_network\n\n" *
            "Hyperparametres :\n- Loss : MSE\n- Optimiser : ADAM\n" *
            "- Batchsize : $batchsize\n" *
            "- Training rate : $training_rate"
        )
    end
end

function custom_train!(loss, ps, data, opt, loss_data, epoch)
    loss_array = Array{Float32,1}(undef, length(data))

    for (i, (x, y)) in enumerate(data)
        loss_array[i] = loss(x, y)
        
        gs = gradient(ps) do
            loss(x, y)
        end
        Flux.update!(opt, ps, gs)
    end

    loss_data[epoch] = mean(loss_array)
end

function nn(gene_layer, cell_layer, molecule_layer, (a, b, c))
    return Chain(
        Flux.Parallel(vcat, gene_layer, cell_layer, molecule_layer),
        Dense(a, b, relu),
        Dense(b, c, relu),
        Dense(c, 1, identity)
    )
end

function train_plot(data, X, Y, (dims1, dims2, dims3), folder, data_matrix, epochs=200; device=gpu)
    println("Creating folder '$(folder)'")
    mkdir("$(output_path)/$(folder)")
    g_dims = dims1
    c_dims = dims2
    m_dims = dims3

    d1_layer = device(Flux.Embedding(length(data_matrix.d1_index), g_dims))
    d2_layer = device(Flux.Embedding(length(data_matrix.d2_index), c_dims))
    d3_layer = device(Flux.Embedding(length(data_matrix.d1_index), m_dims))

    a = dims1 + dims2 + dims3
    b = 50
    c = 10
    
    network = device(nn(d1_layer, d2_layer, d3_layer, (a, b, c)))

    loss(x, y) = Flux.Losses.mse(network(x), y)
    ps = Flux.params(network)
    tr = 0.001
    opt = Flux.ADAM(tr)

    neural_network = ("- Layer 1 : Embeddings\n" * 
        "- Layer 2 : Dense($a, $b, relu)\n" *
        "- Layer 3 : Dense($b, $c, relu)\n" *
        "- Layer 4 : Dense($c, 1, identity)"
    )
    index_file(folder, (g_dims, c_dims, m_dims), data.batchsize, tr, neural_network)

    loss_data = Array{Float32,1}(undef, epochs)

    iter = ProgressBar(1:epochs)
    for i in iter
        custom_train!(loss, ps, data, opt, loss_data, i)

        if mod(i, 10) == 0 || i == 1
            obtained, expected = get_obtained_expected(X, Y, network, device)
            #println(iter, "Prepare intermediate accuracy plot at epoch $(i)...")
            plot_accuracy(folder, i, obtained, expected)
        end
    end

    plot_loss(folder, loss_data)

    if dims1 == 2
        println("Generating dim1 plot...")
        plot_embedding(d1_layer, "dim1", folder, unique(data_matrix.d1_index))
    end
    if dims2 == 2
        println("Generating dim2 plot...")
        plot_embedding(d2_layer, "dim2", folder, unique(data_matrix.d2_index))
    end
    if dims3 == 2
        println("Generating molecule_layer plot...")
        plot_embedding(d3_layer, "molecule_layer", folder, unique(data_matrix.d3_index))
    end
 
    obtained, expected = get_obtained_expected(X, Y, network, device)
    return Statistics.cor(obtained, expected)
end
