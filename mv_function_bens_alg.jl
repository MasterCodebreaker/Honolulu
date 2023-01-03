using LinearAlgebra
using Plots
using Ripserer
using Distances
using Plots
using DataFrames, CSV
using DelimitedFiles
using AbstractAlgebra
using MultivariateStats #for pca
using IterativeSolvers
using NearestNeighbors


"""
Helper functions and global(s)
"""

function drop_simp(simp, collection_of_simps)
    for ss in collection_of_simps
        if ss[3] == simp[1] && ss[2] == simp[2] && ss[1] == simp[3] #CARE reversed here
            return true
        end
    end
    return false
end




function non_zero(alpha)
    B = []
    for i=1:length(alpha)
       if alpha[i] != 0
          push!(B,[i,alpha[i]])
       end
    end
    return B
end

"""
function get_birth(simp_dic_u,simpkey_birth_u, simp)
    for i in 1:length(simp_dic_u)
        if simp_dic_u[i] == simp
            return simpkey_birth_u[i][2]
        end
    end
    print("ERROR", simp)
end
"""



function signed_tetra_volume(a,b,c,d)
    #print(length(a),"HEHE")
    if length(a) == 3
        #print("bad")
        return sign(dot(cross(b-a,c-a),d-a)/6.0)
    end
    #print("good")
    matrixx = zeros(length(a),4)
    matrixx[:,1] = a
    matrixx[:,2] = b
    matrixx[:,3] = c
    matrixx[:,4] = d
    M_00 = fit(PCA, matrixx, maxoutdim = 3)
    pp = predict(M_00, matrixx)
    r = pp[:,1]
    s = pp[:,2]
    t = pp[:,3]
    u = pp[:,4]
    if length(r) < 3 #TOOD
        return -1
    end
    return sign(dot(cross(s-r,t-r),u-r)/6.0)
end

function intersect_line_triangle(q1,q2,triangle, data)
    p1 = [i for i in data[triangle[1]]]
    p2 = [i for i in data[triangle[2]]]
    p3 = [i for i in data[triangle[3]]]
    
    s1 = signed_tetra_volume(q1,p1,p2,p3)
    s2 = signed_tetra_volume(q2,p1,p2,p3)

    if s1 != s2
        s3 = signed_tetra_volume(q1,q2,p1,p2)
        s4 = signed_tetra_volume(q1,q2,p2,p3)
        s5 = signed_tetra_volume(q1,q2,p3,p1)
        if s3 == s4 && s4 == s5
            return true
            #n = np.cross(p2-p1,p3-p1)
            #t = -np.dot(q1,n-p1) / np.dot(q1,q2-q1)
            #return q1 + t * (q2-q1)
        end
    end
    return false
end


function cohomology(data, field)

    """
    We compute second comology group of X
    """
    #threshold = 3
    rips = Ripserer.Rips(data, sparse = true, metric = SqEuclidean())
    collapsed = EdgeCollapsedRips(rips)
    println("Careful if this number is too high: ", length(Ripserer.edges(collapsed)))
    dist = Ripserer.adjacency_matrix(rips)
    diagram_cocycles = ripserer(collapsed; dim_max = 2,reps=true, field = Ripserer.Mod{field})
    dist = Ripserer.distance_matrix(collapsed)
    plot(diagram_cocycles)


    """
    Generator of most persistent 2-coycle which we will map to H^1
    Get second global EPSINLON
    """
    most_persistent_co = diagram_cocycles[3][end]
    #EPSILON = death(most_persistent_co)
    #EPSILON = midlife(most_persistent_co)
    EPSILON = birth(most_persistent_co)
    cocycle = representative(most_persistent_co)

    return dist,EPSILON, cocycle, collapsed

end


function distanced_points(cocycle, dist)

    #Finding the two post distanced points in cloude
    """
    max = 0.0
    index_1 = 0
    index_2 = 0
    for i in 1:length(dist[1,:])
        for j in 1:i
            if dist[i,j] > max
                max = dist[i,j]
                index_1 = i
                index_2 = j
            end
        end
    end
    v_0 = 71
    v_1 = 31
    #v_0,v_1 = 70,30


    """

    # Find vertices on cocycle != 0
    vertices_set = []
    for simp in cocycle
        for p in vertices(simp)
            push!(vertices_set, p)
        end
    end

    # Find most polar of these vertices v_0 v_1, and second most polar that are also distanced from v_0,v_1

    max = 0.0
    v_0 = 0.0
    v_1 = 0.0
    for i in vertices_set
        for j in vertices_set
            if dist[i,j] >= max
                max = dist[i,j]
                v_0 = i
                v_1 = j
            end
        end
    end

    # "Second most polar" that are also distanced from v_0

    u_0 = 0.0
    u_1 = 0.0
    min = 1000
    for j in 1:length(dist[1,:])
        if dist[v_0,j] >= 4* max/5 && dist[v_0,j] <= min
            min = dist[v_0,j]
            u_0 = j
        end
    end

    max = 0.0
    u_1 = 0.0
    
    for j in 1:length(dist[1,:])
        if dist[u_0,j] >= max
            max = dist[u_0,j]
            u_1 = j
        end
    end


    return v_0,v_1,u_0,u_1

end



function piercing_arrays(v_0, v_1, matrix, data, EPSILON,number_of_points, data_dim)

    """
    We want to create two lines, mimenting the normal vector, that penetrates the complex in polar points,
    based on v_0 and v_1
    """
    
    # Finding n-closest nbgh of v_0 and v_1 in order to aproximate normal vector

    number_of_n = number_of_points ÷ 10

    if number_of_n < 10
        number_of_n = 10
    end

    #number_of_n  = 8

    vec_0 = Array([data[v_0][1],data[v_0][2],data[v_0][3]])
    vec_1 = Array([data[v_1][1],data[v_1][2],data[v_1][3]])

    data_matrix = zeros(Float64, (3, length(data)))
    for col in 1:length(data)
        data_matrix[:,col] = [data[col][1],data[col][2],data[col][3]]
    end

    kdtree = KDTree(data_matrix)
    v_0_set, dists = knn(kdtree, vec_0,number_of_n, true)
    v_1_set, dists = knn(kdtree, vec_1,number_of_n, true)



    M_0 = fit(PCA, transpose(matrix[v_0_set,:]), maxoutdim = 3)
    M_1 = fit(PCA, transpose(matrix[v_1_set,:]), maxoutdim = 3)

    #Normals # DO cross product in case the third axis is not shown...
    if data_dim ==3

        axis_0 =  reshape(cross(MultivariateStats.projection(M_0)[:,1], MultivariateStats.projection(M_0)[:,2]),1,3)
        axis_1 = reshape(cross(MultivariateStats.projection(M_1)[:,1], MultivariateStats.projection(M_1)[:,2]),1,3)
    else
        axis_0 = reshape(last([MultivariateStats.projection(M_0)[:,i] for i in 1:3]),1,size(MultivariateStats.projection(M_0))[1])
        axis_1 =reshape(last([MultivariateStats.projection(M_0)[:,i] for i in 1:3]),1,size(MultivariateStats.projection(M_1))[1])
    end 

    # center of mass
    c_0 = sum(matrix[v_0_set[1:3],:], dims = 1)/length(v_0_set[1:3])
    c_1 = sum(matrix[v_1_set[1:3],:], dims = 1)/length(v_1_set[1:3])

    #c_0 = reshape(matrix[v_0,:],1,3)
    #c_1 = reshape(matrix[v_1,:],1,3)


    SCALE_FOR_PIERCING = EPSILON*0.3

    start_0 = c_0 - (SCALE_FOR_PIERCING)*axis_0
    end_0 = c_0 + (SCALE_FOR_PIERCING)*axis_0

    start_1 = c_1 - (SCALE_FOR_PIERCING)*axis_1
    end_1 = c_1 + (SCALE_FOR_PIERCING)*axis_1

    start_0 = reshape(start_0,size(MultivariateStats.projection(M_0))[1])
    end_0 = reshape(end_0,size(MultivariateStats.projection(M_0))[1])
    start_1 = reshape(start_1,size(MultivariateStats.projection(M_0))[1])
    end_1 = reshape(end_1,size(MultivariateStats.projection(M_0))[1])

    return  start_0, start_1, end_0, end_1

end



function get_simplices(collapsed, data, EPSILON,dist, start_0, start_1, end_0, end_1)

    # 1-simplices
    """
    simpkey_birth is of from [Int(key), birth]
    simp_dic is list of tuples of points indexed by keys in simp
    """

    simp_dic = []
    simpkey_birth = []

    simp_1_dic = []
    simpkey_1_birth = []


    u_drop = [] # first coord > 0
    v_drop = [] # first coord < 0


    for i in 1:length(data)
        push!(simp_dic, [i])
        push!(simpkey_birth,[i, 0.0])
    end


    count = length(data)
    for i in 1:length(Ripserer.edges(collapsed))
        birthday = birth(Ripserer.edges(collapsed)[i])
        
        if birthday <= EPSILON# && birthday >= EPSILON*1/2
            count += 1
            push!(simpkey_birth,[count,birthday])
            push!(simp_dic,vertices(Ripserer.edges(collapsed)[i]))
            
            
            push!(simpkey_1_birth,[count,birthday])
            push!(simp_1_dic,vertices(Ripserer.edges(collapsed)[i]))
        """
        elseif vertices(Ripserer.edges(collapsed)[i]) in edges_in_cocycle
            count += 1
            push!(simpkey_birth,[count,birthday])
            push!(simp_dic,vertices(Ripserer.edges(collapsed)[i]))
            
            
            push!(simpkey_1_birth,[count,birthday])
            push!(simp_1_dic,vertices(Ripserer.edges(collapsed)[i]))
        """
        end
    end
    println("number of 1-simps: ", length(simp_1_dic))
    println(" ")


    TT = length(simp_1_dic)
    # 2-simplices
    count = 0
    for i in 1:length(simp_1_dic) # these are closer than epsilon #carefull that we dont add duplicates
        for j in simp_1_dic[i][2]+1:simp_1_dic[i][1]-1
            if dist[simp_1_dic[i][1],j] <= EPSILON && dist[simp_1_dic[i][2],j] <= EPSILON# && dist[simp_1_dic[i][1],j] >= EPSILON*1/4
                
                index = [simp_1_dic[i][2],j,simp_1_dic[i][1]]
                
                if intersect_line_triangle(start_0,end_0,index, data)
                    push!(u_drop, (index[1],index[2],index[3]))
                elseif intersect_line_triangle(start_1,end_1,index, data)
                    push!(v_drop, (index[1],index[2],index[3]))
                else
                    #pass
                end
                count += 1
                push!(simp_dic, (index[3],index[2],index[1]))
            end
        end
    end

    println("number of all simps: ",length(simp_dic))
    

    return simp_dic, u_drop, v_drop, TT
end




function restriction_alpha_u_alpha_v(cocycle, u_drop,v_drop, EPSILON)

    """
    Restricting alpha to alpha_U and alpha_V
    """


    cocycle_u = []
    for simp in cocycle
        ok = true
        if length(vertices(simp)) > 2
            if vertices(simp) in u_drop
                println(vertices(simp), "IS THIS RIGHT?")
                ok = false
            else
                #ok = true
                #pass
            end
        end
        if ok && birth(simp) <= EPSILON
            push!(cocycle_u, simp)
            
        end
    end
    


    cocycle_v = []
    for simp in cocycle
        ok = true
        if length(vertices(simp)) > 2
            if vertices(simp) in v_drop
                ok = false
            else
                #ok = true
                #pass
            end
        end
        if ok && birth(simp) <= EPSILON
            push!(cocycle_v, simp)
            
        end
    end
    
    return cocycle_u, cocycle_v
end



function computing_differential(simp_dic,cocycle_u, cocycle_v, u_drop, v_drop, number_of_points, F, field, data, len_simp_1_dic)
    D = zeros(Int64,length(simp_dic),length(simp_dic))
    alpha_u = zeros(Int64,length(simp_dic))
    alpha_u = reshape(alpha_u, length(simp_dic), 1)

    zero_col_u =[]
    zero_col_v =[]

    alpha_v = zeros(Int64,length(simp_dic))
    alpha_v = reshape(alpha_v, length(simp_dic), 1)

    indexx = transpose([[1,2]  [1,3]  [2,3]])

    sorted_1_simps = []

    for col in 1:length(simp_dic)
        simplex = simp_dic[col]#simp_dic[Int(sort_ar_simp[col,1])]
        #print(simplex)
        
        if length(simplex) > 2 
            if drop_simp(simplex,u_drop)
                #print("ok_u")
                push!(zero_col_u, col)
            end
        end
        
        ok_v = false
        if length(simplex) > 2 
            if drop_simp(simplex,v_drop)
                #print("ok_v")
                push!(zero_col_v, col)
            end
        end

        if length(simplex) == 1
            #pass
        elseif length(simplex) == 2
            push!(sorted_1_simps, simplex)
            D[Int64(simplex[1]), col] = 1
            D[Int64(simplex[2]), col] = -1
            
        elseif length(simplex) == 3
            #println(simplex,"--", cocycle_u[1])
            for c in cocycle_u
                if vertices(c) == (Int64(simplex[1]),Int64(simplex[2]),Int64(simplex[3]))
                    alpha_u[col] = Int64(coefficient(c))
                    
                end
            end
            
            for c in cocycle_v
                if vertices(c) == (Int64(simplex[1]),Int64(simplex[2]),Int64(simplex[3]))
                    alpha_v[col] = Int64(coefficient(c))
                    
                end
            end
            
            indexes = []
            for arr in 1:length(sorted_1_simps)
                for ind in 1:3
                    x = indexx[ind,1]
                    y = indexx[ind,2]
                    if sorted_1_simps[arr] == (simplex[x],simplex[y])
                        push!(indexes, arr+number_of_points)
                    end
                end
            end
            if length(indexes) == 3
                
                D[indexes[1], col] = 1
                D[indexes[2], col] = -1
                D[indexes[3], col] = 1
            else
                #println("ERROR, something wrong with the order", simplex, "..", indexes)
            end
        

        end
    end
    #return D, alpha_u, alpha_v, zero_col_u, zero_col_v

    zero_col_u_v = vcat(zero_col_u, zero_col_v)
    D_u = deepcopy(D)
    D_v = deepcopy(D)
    D_u_v = deepcopy(D)
    for i in zero_col_u
        D_u[:,i] = zeros(length(simp_dic))
        D_u[i,:] = zeros(length(simp_dic))
    end
    for i in zero_col_v
        D_v[:,i] = zeros(length(simp_dic))
        D_u[i,:] = zeros(length(simp_dic))
    end
    for i in zero_col_u_v
        D_u_v[:,i] = zeros(length(simp_dic))
        D_u[i,:] = zeros(length(simp_dic))
    end

    D_u_T = transpose(D_u)
    M_u = AbstractAlgebra.matrix(F,D_u_T)
    b_u = AbstractAlgebra.matrix(F,alpha_u)

    D_v_T = transpose(D_v)
    M_v = AbstractAlgebra.matrix(F,D_v_T)
    b_v = AbstractAlgebra.matrix(F,alpha_v)

    gamma_u = AbstractAlgebra.solve(M_u, b_u)
    gamma_v = AbstractAlgebra.solve(M_v, b_v)

    # Restrict 

    for i in 1:length(gamma_u)
        if i in zero_col_u_v
            gamma_u[i,1] = 0
        end
    end

    for i in 1:length(gamma_v)
        if i in zero_col_u_v
            gamma_v[i,1] = 0
        end
    end

    x = gamma_u - gamma_v

    B = Array{Float64}(undef, size(x)[1])

    for i in 1:size(x)[1]
        B[i] = AbstractAlgebra.lift(deepcopy(x[i]))
    end

    for i in 1:length(B)
        if B[i] >= field÷2
            B[i] -= field
        end
    end


    ddd = zeros(Int64, size(D_u_v))
    D_u_v_T = transpose(D_u_v)
    for i in 1:size(D_u_v_T)[1]
        for j in 1:size(D_u_v_T)[1]        
            ddd[i,j] = Int64(D_u_v_T[i,j])
        end
    end

    lenn = length(data)+ len_simp_1_dic
    beta_new = B[1:lenn]
    ddd_new = ddd[1:lenn,1:lenn]
    b = IterativeSolvers.lsmr(ddd_new, beta_new)


    theta = b[1:number_of_points]
    return theta

end


function mv(; double_penetration::Bool = false,
    tripple::Bool = false,
    start_points::Bool = false,
    points::Array = [0,0,0,0],
    input_file::String="./cool_big_data.csv",
    anti_podes::String="./anti_podes",
    output_file::String="./point_color",
    penetrating_vectors::String="./vectors",
    naive_penetration::Bool=false )
    field = 47
    F = AbstractAlgebra.GF(field)

    matrix = readdlm(input_file, ',', Float64)
    data = []
    for row in 1:length(matrix[:,1])
        arr = [matrix[row,i] for i in 1:length(matrix[1,:])]
        push!(data, arr)
    end

    data_dim = length(data[1])
    number_of_points = length(data)
    println("Number of points: ", number_of_points, " ")

    dist,EPSILON, cocycle, collapsed = cohomology(data, field)

    """
    Get 1-simplecis building up the 2-simplecies in cocycle
    """

    edges_in_cocycle = []
    for simp in cocycle
        push!(edges_in_cocycle, (vertices(simp)[1],vertices(simp)[2]))
        push!(edges_in_cocycle, (vertices(simp)[1],vertices(simp)[3]))
        push!(edges_in_cocycle, (vertices(simp)[2],vertices(simp)[3]))
    end
    edges_in_cocycle = unique(edges_in_cocycle)

    if start_points
        if tripple
            v_0,v_1,u_0,u_1,w_0,w_1 = points[1], points[2], points[3], points[4] , points[5], points[6]
        else
            v_0,v_1,u_0,u_1 = points[1], points[2], points[3], points[4]
            w_0, w_1 = 0,0
        end
        #v_0,v_1,u_0,u_1 = points[1], points[2], points[3], points[4]
    else
        v_0,v_1,u_0,u_1 = distanced_points(cocycle, dist)

    end

    df = DataFrame(
    v_0=v_0,
    v_1=v_1,
    u_0=u_0,
    u_1=u_1,
    w_0 = w_0,
    w_1 = w_1 
    )
    CSV.write(anti_podes * ".csv", df)

    # Do this twice
    if naive_penetration

        start_0 = [0.0,0.0,1.0+EPSILON]
        start_1 = [0.0,0.0,-1.0+EPSILON]
        end_0 = [0.0,0.0,1.0-EPSILON]
        end_1 = [0.0,0.0,-1.0-EPSILON]
    else
        start_0, start_1, end_0, end_1 = piercing_arrays(v_0, v_1, matrix, data, EPSILON,number_of_points,data_dim)
    end
    #start_0, start_1, end_0, end_1 = piercing_arrays(v_0, v_1, matrix, data, EPSILON,number_of_points,data_dim)
    simp_dic, u_drop, v_drop, len_simp_1_dic  = get_simplices(collapsed, data, EPSILON,dist, start_0, start_1, end_0, end_1)
    cocycle_u, cocycle_v = restriction_alpha_u_alpha_v(cocycle, u_drop,v_drop, EPSILON)
    theta = computing_differential(simp_dic,cocycle_u, cocycle_v, u_drop, v_drop, number_of_points,F, field,data, len_simp_1_dic)

    # Write to file

    x = [x[1] for x in data]
    y = [x[2] for x in data]
    z = zeros(Float64,length(x))
    z = [x[3] for x in data]
    c = theta
    df = DataFrame(
        x=x,
        y=y,
        z=z,
        c=c
    )
    CSV.write(output_file * ".csv", df)


    df = DataFrame(
    x=start_0,
    y=end_0,
    z=start_1,
    c=end_1
    )
    CSV.write(penetrating_vectors * ".csv", df)

    #println("First penetration done.")

    if double_penetration
        println("Starting second penetration")
        #start_0, start_1, end_0, end_1 = piercing_arrays(u_0, u_1, matrix, data, EPSILON,number_of_points,data_dim)
        if naive_penetration
            start_0 = [0.0,1.0+EPSILON,0.0]
            start_1 = [0.0,-1.0+EPSILON,0.0]
            end_0 = [0.0,1.0-EPSILON,0.0]
            end_1 = [0.0,-1.0-EPSILON,0.0]
        else
            start_0, start_1, end_0, end_1 = piercing_arrays(u_0, u_1, matrix, data, EPSILON,number_of_points,data_dim)
        end

        simp_dic, u_drop, v_drop, len_simp_1_dic  = get_simplices(collapsed, data, EPSILON,dist, start_0, start_1, end_0, end_1)
        cocycle_u, cocycle_v = restriction_alpha_u_alpha_v(cocycle, u_drop,v_drop, EPSILON)
        theta = computing_differential(simp_dic,cocycle_u, cocycle_v, u_drop, v_drop, number_of_points,F, field,data, len_simp_1_dic)


        x = [x[1] for x in data]
        y = [x[2] for x in data]
        z = zeros(Float64,length(x))
        z = [x[3] for x in data]
        c = theta
        df = DataFrame(
            x=x,
            y=y,
            z=z,
            c=c
        )
        CSV.write(output_file * "double" * ".csv", df)
    
    
        df = DataFrame(
        x=start_0,
        y=end_0,
        z=start_1,
        c=end_1
        )
        CSV.write(penetrating_vectors * "double" * ".csv", df)

        if tripple
            println("Starting third penetration")
            #start_0, start_1, end_0, end_1 = piercing_arrays(u_0, u_1, matrix, data, EPSILON,number_of_points,data_dim)
            if naive_penetration
                start_0 = [0.0,1.0+EPSILON,0.0]
                start_1 = [0.0,-1.0+EPSILON,0.0]
                end_0 = [0.0,1.0-EPSILON,0.0]
                end_1 = [0.0,-1.0-EPSILON,0.0]
            else
                start_0, start_1, end_0, end_1 = piercing_arrays(w_0, w_1, matrix, data, EPSILON,number_of_points,data_dim)
            end
    
            simp_dic, u_drop, v_drop, len_simp_1_dic  = get_simplices(collapsed, data, EPSILON,dist, start_0, start_1, end_0, end_1)
            cocycle_u, cocycle_v = restriction_alpha_u_alpha_v(cocycle, u_drop,v_drop, EPSILON)
            theta = computing_differential(simp_dic,cocycle_u, cocycle_v, u_drop, v_drop, number_of_points,F, field,data, len_simp_1_dic)
    
    
            x = [x[1] for x in data]
            y = [x[2] for x in data]
            z = zeros(Float64,length(x))
            z = [x[3] for x in data]
            c = theta
            df = DataFrame(
                x=x,
                y=y,
                z=z,
                c=c
            )
            CSV.write(output_file * "tripple" * ".csv", df)
        
        
            df = DataFrame(
            x=start_0,
            y=end_0,
            z=start_1,
            c=end_1
            )
            CSV.write(penetrating_vectors * "tripple" * ".csv", df)
        end

    end

    
    println("All penetration done.")
    #return v_0,v_1,u_0,u_1, tripple

end
