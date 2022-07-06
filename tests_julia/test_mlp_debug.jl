using CSV 
using DataFrames
using Dates
using CUDA
using Flux
using Statistics
using Random
using Gadfly
using ProgressBars
using Cairo

basepath = "/u/sauves/leucegene-shared"
outpath  = "./RES/EMBEDDINGS" # our output directory
outdir = "$(outpath)/embeddings_$(now())"
mkdir(outdir)
counts_file = "$(basepath)/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv"
#GE_TRSC_TPM = DataFrame(CSV.File(filename))
@time GE_CDS_TPM = CSV.read(counts_file, DataFrame)
clinical_f_file = "$(basepath)/Data/LEUCEGENE/lgn_pronostic_CF"
CF = CSV.read(clinical_f_file, DataFrame)

struct Data
    name::String
    data::Array
    factor_1::Array
    factor_2::Array
end

struct FoldData 
    name::String 
    train::Data
    test::Data
end 

device!(1)
index = GE_CDS_TPM[:,1]
data_full = GE_CDS_TPM[:,2:end] 
cols = names(data_full)
# log transforming
data_full = log10.(Array(data_full) .+ 1)
# remove least varying genes
ge_var = var(data_full,dims = 1) 
ge_var_med = median(ge_var)
# high variance only 
hvg = getindex.(findall(ge_var .> ge_var_med),2)[1:Int(floor(end/10))]
data_full = data_full[:,hvg]
cols = cols[hvg]
# verify that the operation worked
sum(var(data_full, dims = 1) .< ge_var_med)
# split into train test
nsamples = length(index)
indices = shuffle(Array{Int}(1:nsamples))
nfolds = 5
foldsize = Int(nsamples / 5)
# training metavariables
nepochs = 10_000
tr = 0.001
wd = 1e-3
patient_emb_size = 2
gene_emb_size = 50 


folds = Array{FoldData}(undef,5)
for i in 1:nfolds
    tst_idx = indices[(i - 1) * foldsize + 1: i * foldsize]
    tr_idx = setdiff(indices, tst_idx)
    test = Data("test", data_full[tst_idx,:], index[tst_idx], cols)
    train = Data("train", data_full[tr_idx,:], index[tr_idx], cols)
    fold_data = FoldData("fold_$i", train, test)
    folds[i] = fold_data
end
function l2_penalty(model)
    penalty = 0
    for layer in model[2:end]
        if typeof(layer) != typeof(vec) && typeof(layer) != typeof(Flux.Parallel)
            penalty += sum(abs2, layer.weight)
        end
    end
    return penalty
end

function plot_accuracy(epoch, obtained, expected)
    corP = Statistics.cor(obtained, expected)
    draw(
        PNG("$(outdir)/accuracy_at_epoch_$epoch.png"), 
        plot(
            x=expected, y=obtained, Guide.title("Accuracy at epoch $epoch (r = $corP)"), 
            Guide.xlabel("Expected"), Guide.ylabel("Obtained"), Geom.abline, 
            Geom.hexbin(xbincount=100, ybincount=100)
        )
    )
end

# function prep_data(data::Array)
#     # create network
#     nsamples = size(data)[1]
#     ngenes = size(data)[2]
#     mat = rand(nsamples, ngenes)

#     insize_f1 = nsamples
#     insize_f2 = ngenes 
    
#     #X = Array{Int32, 2}(undef, insize_f1 * insize_f2)
#     f1 = collect(1:insize_f1)
#     f2 = collect(1:insize_f2)
#     #f1 = Array{Int32}(1:insize_f1)
#     #f2 = Array{Int32}(1:insize_f2)

#     X_1 = gpu(vcat(Array{Int32}(f1 * transpose(ones(1:insize_f2)))...)) 
#     X_2 = gpu(vcat(transpose(Array{Int32}(f2 * transpose(ones(1:insize_f1))))...))
#     Y = gpu(vcat(mat...))
#     return ((X_1, X_2), Y)
# end 

function prep_data(data::Data; device = gpu)
    ## data preprocessing
    ### remove index columns, log transform
    n = length(data.factor_1)
    m = length(data.factor_2)
    values = Array{Float32,2}(undef, (1, n * m))
    #print(size(values))
    factor_1_index = Array{Int32,1}(undef, n * m)
    factor_2_index = Array{Int32,1}(undef, n * m)
     # d3_index = Array{Int32,1}(undef, n * m)
    
    for i in 1:n
        for j in 1:m
            index = (i - 1) * m + j 
            values[1, index] = data.data[i, j]
            factor_1_index[index] = i # Int
            factor_2_index[index] = j # Int 
            # d3_index[index] = data.d3_index[i] # Int 
        end
    end
    return (device(factor_1_index), device(factor_2_index)), device(vec(values))
end

function generate_2D_embedding(data::Data)
    X_, Y_ = prep_data(data)
    model = nn(patient_emb_size, gene_emb_size, length(data.factor_1), length(data.factor_2))
    loss_array = Array{Float32, 1}(undef, nepochs)
    opt = Flux.ADAM(tr)

    @time for e in ProgressBar(1:nepochs)
        # out = model(X_)
        ps = Flux.params(model)
        loss_array[e] = loss(X_, Y_, model)
        gs = gradient(ps) do 
            loss(X_, Y_, model)
        end
        Flux.update!(opt, ps, gs)
    end
    
    accuracy = cor(cpu(model(X_)), cpu(Y_))
    
    plot_accuracy(nepochs, cpu(model(X_)), cpu(Y_))
    return cpu(transpose(model[1][1].weight)), loss_array, accuracy
end

# fold_data = folds[1]
# X_, Y_ = prep_data(fold_data.train.data)
# nsamples = length(fold_data.train.factor_1)
# ngenes = length(fold_data.train.factor_2)


function nn(emb_size_1::Int, emb_size_2::Int, f1_size::Int, f2_size::Int) 
    a = emb_size_1 + emb_size_2
    b, c = 50, 10
    emb_layer_1 = gpu(Flux.Embedding(f1_size, emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(f2_size, emb_size_2))
    return gpu(Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        Dense(a, b, relu), 
        Dense(b, c, relu),
        Dense(c, 1, identity),
        vec
    ))
end

loss(x, y)= Flux.Losses.mse(model(x), y)
loss(x, y, model) = Flux.Losses.mse(model(x), y) + l2_penalty(model) * wd
# loss(X_, Y_)

data_struct = Data("full", data_full, index, cols)
embed_1, losses_1, acc_1= generate_2D_embedding(data_struct)
embed_2, losses_2, acc_2 = generate_2D_embedding(data_struct)

println("accuracy_1 $acc_1\naccuracy_2: $acc_2")

df1 = DataFrame(Dict([("emb$i", embed_1[:,i]) for i in 1:size(embed_1)[2]]))
df2 = DataFrame(Dict([("emb$i", embed_2[:,i]) for i in 1:size(embed_2)[2]]))
df1.index = index
df2.index = index
df1.group1 = CF[:,"Cytogenetic risk"]
df2.group1 = CF[:,"Cytogenetic risk"]
df1.group2 =  CF[:,"WHO classification"] 
df2.group2 =  CF[:,"WHO classification"]
CSV.write("$outdir/output_embedding_1", df1)
CSV.write("$outdir/output_embedding_2", df2)

# graph_1 = plot(x=embed_1[:, 1], y=embed_1[:, 2], 
#     label=index, Geom.point, Geom.label, 
#     ,
#     Coord.cartesian(xmin=-5, xmax=5, ymin=-5, ymax=5)
# )
# graph_2 = plot(x=embed_2[:, 1], y=embed_2[:, 2], 
#     label=index, Geom.point, Geom.label, 
#     Theme(panel_fill="white"),
#     Coord.cartesian(xmin=-5, xmax=5, ymin=-5, ymax=5)
# )
# binsize =  10
# graph1 = plot_embedding(embed_1, nepochs, binsize)
# graph2 = plot_embedding(embed_2, nepochs, binsize)
# #df = DataFrame(losses_1 = losses_1, losses_2 = losses_2, epoch=1:nepochs)
# #graph_loss = plot(stack(df, [:losses_1, :losses_2]), x=:epoch, y=:value, 
#         color=:variable, Guide.xlabel("Epoch"), Guide.ylabel("Loss"), 
#         Guide.title("Loss per epoch"), Theme(panel_fill="white"), Geom.line)

# draw(SVG("$(outdir)/graph_embed_1.svg"), graph1)
# draw(PNG("$(outdir)/graph_embed_1.png"), graph1)

# draw(SVG("$(outdir)/graph_embed_2.svg"), graph2)
# draw(PNG("$(outdir)/graph_embed_2.png"), graph2)

# draw(SVG("$(outdir)/graph_loss.svg"), graph_loss)