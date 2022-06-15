using DelimitedFiles
using DataFrames

function scatterplot(x, y, color, title::String="Title", xlabel::String="x", ylabel::String="y", 
    color_title::String="Color", draw_png=true)

    p = plot(x=x, y=y, Geom.point, Geom.abline(), Guide.title(title), Guide.xlabel(xlabel), Guide.ylabel(ylabel), 
    color=color, Guide.colorkey(title=color_title));

    return draw_png ? draw(PNG(), p) : p
end

function scatterplot(x, y, title::String="Title", xlabel::String="x", ylabel::String="y", draw_png=true)

    p = plot(x=x, y=y, Geom.point, Geom.abline(), Guide.title(title), Guide.xlabel(xlabel), Guide.ylabel(ylabel));

    return draw_png ? draw(PNG(), p) : p
end

struct Parseddata
    data::Array{Float32,2}
    data_log::Array{Float32,2}
    id::Array{String,1}
    gene_symbol::Array{String,1}
    CL_Name::Array{String,1}
    SM_Dose::Array{Float32,1}
    SM_Name::Array{String,1}
end

function parser(file::String)
    raw_data = DelimitedFiles.readdlm(file, '\t', Any, '\n')
    
    data = Array{Float32,2}(raw_data[15:end,13:end])
    data_log = log10.(data .+ 1)
    
    CL_Name = Array{String,1}(raw_data[4,13:end])
    SM_Dose = Array{Float32,1}(raw_data[7,13:end])
    SM_Name = Array{String,1}(raw_data[10,13:end])
    
    id = Array{String,1}(raw_data[15:end,1])
    gene_symbol = Array{String,1}(raw_data[15:end,8])
    
    return Parseddata(data, data_log, id, gene_symbol, CL_Name, SM_Dose, SM_Name)
end

function dataframe_narrow(data::Parseddata, log::Bool=true)
    m = length(data.CL_Name)
    n = length(data.gene_symbol)
    df = DataFrame(CL_Name=Array{String,1}(undef, m * n), SM_Dose=Array{Float32,1}(undef, m * n), 
        SM_Name=Array{String,1}(undef, m * n), gene=Array{String,1}(undef, m * n), expr=Array{Float32,1}(undef, m * n))
    
    for i in 1:m
        for j in 1:n
            index = (i - 1) * n + j
            df.CL_Name[index] = data.CL_Name[i]
            df.SM_Dose[index] = data.SM_Dose[i]
            df.SM_Name[index] = data.SM_Name[i]
            df.gene[index] = data.gene_symbol[j]
            if log
                df.expr[index] = data.data_log[j, i]
            else
                df.expr[index] = data.data[j, i]
        end
        end   
    end
    
    return df
end