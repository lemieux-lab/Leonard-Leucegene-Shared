using Pkg
using CSV 
using DataFrames
using Dates
using CUDA
using Flux
using Statistics
using Random
using ProgressBars
filename = "/u/sauves/leucegene-shared/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv"
#GE_TRSC_TPM = DataFrame(CSV.File(filename))
@time GE_CDS_TPM = CSV.read(filename, DataFrame)
print()
mutable struct Data
    name::String
    data::Array
    factor_1::Array
    factor_2::Array
end 
mutable struct FoldData 
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
hvg = getindex.(findall(ge_var .> ge_var_med),2)
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
folds[1].train.factor_1
folds[1].train.data




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
    return (device(factor_1_index), device(factor_2_index)), device(values)
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

function nn(factor_1_layer, factor_2_layer, (a, b, c))
    return Chain(
        Flux.Parallel(vcat, factor_1_layer, factor_2_layer),
        Dense(a, b, relu),
        Dense(b, c, relu),
        Dense(c, 1, identity)
    )
end
function train_embeddings(dl, X_, Y_, (dims1, dims2), folder, data, epochs = 200; device = gpu)
    #prepare outpath
    mkdir("$(folder)")
    # construct embeddings & the neural net
    d1_layer = device(Flux.Embedding(length(data.factor_1), dims1))
    d2_layer = device(Flux.Embedding(length(data.factor_2), dims2))
    
    a = dims1 + dims2 
    b = 50
    c = 10 

    network = device(nn(d1_layer, d2_layer, (a, b, c)))

    loss(x, y) = Flux.Losses.mse(network(x), y)
    ps = Flux.params(network)
    tr = 0.001
    opt = Flux.ADAM(tr)

    neural_network = ("- Layer 1 : Embeddings\n" * 
        "- Layer 2 : Dense($a, $b, relu)\n" *
        "- Layer 3 : Dense($b, $c, relu)\n" *
        "- Layer 4 : Dense($c, 1, identity)"
    )
    println(neural_network)
    loss_data = Array{Float32,1}(undef, epochs)
    # cycle through epochs
    for e in ProgressBar(1:epochs)
        custom_train!(loss, ps, dl, opt, loss_data, e)
    end
    println("final loss: $(loss_data[end])")

end

outpath = "output"
mb_size = 2 ^ 14
fold_data = folds[1]

X_tr, Y_tr = prep_data(fold_data.train)
train_dl =  Flux.Data.DataLoader((X_tr, Y_tr), batchsize = mb_size)
X_tst, Y_tst = prep_data(fold_data.test)
test_dl = Flux.Data.DataLoader((X_tst, Y_tst), batchsize = mb_size)
#println(size(X_tr[1]))
#println(size(X_tst[1]))
# train embedding, retrieve model
train_embeddings(train_dl, X_tr, Y_tr, (2,2), "$(outpath)/embeddings_$(now())", fold_data.train, 500)

