using CSV 
using DataFrames
using Dates
using CUDA
using Flux
using Statistics
using Random
using ProgressBars
using Flux: params 
# Load data
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
tr = 1e-3
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

function prep_data(data::Data; device = gpu)
    ## data preprocessing
    ### remove index columns, log transform
    n = length(data.factor_1)
    m = length(data.factor_2)
    values = Array{Float32,2}(undef, (1, n * m))
    #print(size(values))
    factor_1_index = Array{Int32,1}(undef, max(n * m, 1))
    factor_2_index = Array{Int32,1}(undef, max(n * m, 1))
     # d3_index = Array{Int32,1}(undef, n * m)
    
    while i <= n
        for j in 1:m
            index = (i - 1) * m + j 
            values[1, index] = data.data[i, j]
            factor_1_index[index] = i # Int
            factor_2_index[index] = j # Int 
            # d3_index[index] = data.d3_index[i] # Int 
        end
        i+=1
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
    
    return model, cpu(transpose(model[1][1].weight)), loss_array, accuracy

end
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

data_struct = Data("full", data_full, index, cols)
# Generate embeddings 
model, embed, losses, acc = generate_2D_embedding(data_struct)
df1 = DataFrame(Dict([("emb$i", embed[:,i]) for i in 1:size(embed)[2]]))
df1.index = index
df1.group1 = CF[:,"Cytogenetic risk"]
df1.group2 =  CF[:,"WHO classification"] 

# choose random index 
left_out = Int(round((rand(1) * 300)[1]))
# Inspect 2d patient embedding
df1.group3 = map(Int, zeros(size(df1)[1]))
df1.group3[left_out] = 1
CSV.write("$outdir/output_embedding_1", df1)

function interpolate(data_struct, grid_points;nepochs_test=100, tr_test=0.001)
    X_, Y_ = prep_data(data_struct)
    opt = Flux.Optimise.ADAM(tr_test)
   
    traject_x = zeros(nepochs_test * length(grid_points))
    traject_y = zeros(nepochs_test * length(grid_points))
    init_pos = Array{String}(undef, nepochs_test * length(grid_points))

    for init_n in ProgressBar(1:length(grid_points))
        # copy the trained model 
        interp_model = nn(patient_emb_size, gene_emb_size, nsamples,length(cols) )
        Flux.loadparams!( interp_model, params(model))
        # Random init embedding for selected sample
        init_coord = (grid_points[init_n][1], grid_points[init_n][2])
        params(interp_model)[1][:,left_out] = init_coord
        # Run interpolation
        for e in 1:nepochs_test
            offset = nepochs_test * (init_n - 1)
            traject_x[e + offset] = cpu(interp_model[1][1].weight')[left_out,1]
            traject_y[e + offset] = cpu(interp_model[1][1].weight')[left_out,2]
            init_pos[e + offset] = "$init_coord"
            ps = params(interp_model[1][1]) # Freeze other parameters of network, only emb1 layer can move
            gs = gradient(() -> loss(X_, Y_, interp_model), ps)
            Flux.update!(opt, ps, gs)
            #println(loss(X_, Y_, model))
        end
    end
    #init_pos = vec((ones(length(grid_points), nepochs_test) .+ collect(0:length(grid_points)-1))')

    # trajectories 
    df = DataFrame(Dict([("init_pos", init_pos), ("emb1", traject_x ),("emb2", traject_y)]))
    CSV.write("$outdir/interpolation_trajectories", df)
end 
function generate_grid_points(xmin, xmax, ymin,ymax;res=40)
    x = ones(res, res) .+ LinRange(xmin-1,xmax-1,res)
    y = ones(res, res) .+ LinRange(ymin-1,ymax-1,res)
    grid_points = [(x,y) for (x,y) in zip(x, y')]
    return grid_points
end
grid_points = generate_grid_points(-10,10,-10,10,res=10)
function filter_data(data_struct, left_out)
    new_data_struct = Data("left_out",
    data_struct.data[left_out,:]',
    [data_struct.factor_1[left_out]],
    data_struct.factor_2,)
    return new_data_struct
end

interpolate(data_struct, grid_points, nepochs_test = 2_000)

println(loss(X_, Y_, model))
embed_layer_test = cpu(model[1][1].weight')
df3 =  DataFrame(Dict([("emb$i", embed_layer_test[:,i]) for i in 1:size(embed_layer_test)[2]]))
df3.index = index
df3.group1 = CF[:,"Cytogenetic risk"]
df3.group2 =  CF[:,"WHO classification"] 
df3.group3 = map(Int, zeros(size(df3)[1]))
df3.group3[left_out] = 1
# Inspect final position of interpolated sample
CSV.write("$outdir/output_embedding_1_interpolated", df3)

