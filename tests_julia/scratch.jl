using CUDA
using Flux
x = [1,2,3]
x =[1 2;3 4]
x = rand(Float32, 5,3)
W = rand(5,10)
x = rand(10)
f(x) = 3*x^2 + 2x + 1
f(5)
df(x) = gradient(f, x)[1]
df(5)

mysin(x) = sum((-1)^k*x^(1+2k)/factorial(1+2k) for k in 0:5)
x = 0.5

mysin(x), gradient(mysin,x)[1]
sin(x), cos(x)
myloss(W, b, x) = sum(W * x .+ b)
W = randn(5,7)
b = zeros(5)
x = rand(7)
# basic syntax (function, (parameters))
gradient(myloss, W, b, x)

using Flux: params
nnodes = 4
input_dim = 12
W = randn(nnodes, input_dim)
b = zeros(nnodes)
x = rand(input_dim)
y1(x) = sum(W * x .+ b)
# other syntax using anon. fn 
grads = gradient(()->y1(x), params([W, b]))
grads[b]
# other syntax


m = Dense(10,5)
x = rand(Float32, 10)
m(x)
params(m)
m = Chain(Dense(10,5, relu), Dense(5,2), softmax)
l(x) = sum(Flux.crossentropy(m(x), [0.5,0.5]))
params(m)
# final syntax 
grads = gradient(params(m)) do 
    l(x)
end
for p in params(m)
    println(grads[p])
end 
using Flux.Optimise: update!, Descent
mu = 0.1
for p in params(m)
  p .-= mu * grads[p]
end
# which is totally equivalent to
opt = Descent(mu)
for p in params(m)
    update!(opt, p, grads[p])
end

# but very useful when using a more complex optimizer such as adam 
opt = Flux.Optimise.ADAM(mu)
for p in params(m)
    update!(opt, p, grads[p])
end 

loss(x,y) = sum(Flux.crossentropy(m(x), y))

for i in 1:5
    data, labels = rand(10,100), fill(0.5,2,100)
    gs = gradient(params(m)) do
        l = loss(data, labels) 
    end
    update!(opt, params(m), gs)
end

# challenge: write gradient function to this model
W = rand(2, 5)
b = rand(2)

predict(x) = W*x .+ b

function loss(x, y)
  ŷ = predict(x)
  sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2) # Dummy data
loss(x, y)
