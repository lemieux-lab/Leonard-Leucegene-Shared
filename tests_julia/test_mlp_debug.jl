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

filename = "/u/sauves/leucegene-shared/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv"
#GE_TRSC_TPM = DataFrame(CSV.File(filename))
@time GE_CDS_TPM = CSV.read(filename, DataFrame)

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

index = GE_CDS_TPM[:,1]
data = GE_CDS_TPM[:,2:end] 
cols = names(data)
# log transforming
data = log10.(Array(data) .+ 1)
# remove least varying genes
ge_var = var(data,dims = 1) 
ge_var_med = median(ge_var)
# high variance only 
hvg = getindex.(findall(ge_var .> ge_var_med),2)[1:Int(floor(end/10))]
data = data[:,hvg]
cols = cols[hvg]
# verify that the operation worked
sum(var(data, dims = 1) .< ge_var_med)
# split into train test
nsamples = length(index)
indices = shuffle(Array{Int}(1:nsamples))
nfolds = 5
foldsize = Int(nsamples / 5)

folds = Array{FoldData}(undef,5)
for i in 1:nfolds
    tst_idx = indices[(i - 1) * foldsize + 1: i * foldsize]
    tr_idx = setdiff(indices, tst_idx)
    test = Data("test", data[tst_idx,:], index[tst_idx], cols)
    train = Data("train", data[tr_idx,:], index[tr_idx], cols)
    fold_data = FoldData("fold_$i", train, test)
    folds[i] = fold_data
end


function prep_data(data::Array)
    # create network
    nsamples = size(data)[1]
    ngenes = size(data)[2]
    mat = rand(nsamples, ngenes)

    insize_f1 = nsamples
    insize_f2 = ngenes 
    
    #X = Array{Int32, 2}(undef, insize_f1 * insize_f2)
    f1 = collect(1:insize_f1)
    f2 = collect(1:insize_f2)
    #f1 = Array{Int32}(1:insize_f1)
    #f2 = Array{Int32}(1:insize_f2)

    X_1 = gpu(vcat(Array{Int32}(f1 * transpose(ones(1:insize_f2)))...)) 
    X_2 = gpu(vcat(transpose(Array{Int32}(f2 * transpose(ones(1:insize_f1))))...))
    Y = gpu(vcat(mat...))
    return ((X_1, X_2), Y)
end 

emb_size_1 = 2
emb_size_2 = 2


a = emb_size_1 + emb_size_2
b, c = 50, 10
nepochs = 2000
tr = 0.001
opt = Flux.ADAM(tr)


fold_data = folds[1]
X_, Y_ = prep_data(fold_data.train)
nsamples = length(fold_data.train.factor_1)
ngenes = length(fold_data.train.factor_2)
emb_layer_1 = gpu(Flux.Embedding(nsamples, emb_size_1))
emb_layer_2 = gpu(Flux.Embedding(ngenes, emb_size_2))
model = gpu(Chain(
Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
Dense(a, b, relu), 
Dense(b, c, relu),
Dense(c, 1, identity)))

loss = Flux.Losses.mse(model(x), y)

loss_array = Array{Float32, 1}(undef, nepochs)
loss_array

@time for e in ProgressBar(1:nepochs)
    # out = model(X_)
    ps = Flux.params(model)
    loss_array[e] = loss(X_, Y_)
    gs = gradient(ps) do 
        loss(X_, Y_)
    end
    Flux.update!(opt, ps, gs)
end
loss_array
# forward prop 
