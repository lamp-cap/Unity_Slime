using Unity.Mathematics;

namespace Slime
{
    public class Eigen
    {
        private const float EPSILON = 1e-6f;
        public static void EigenSymmetric3X3(float3x3 symCov, out float3 lambda, out float3x3 V)
        {
            float3x3 A = symCov;

            float m = (A.c0.x + A.c1.y + A.c2.z) / 3.0f;
            A.c0.x -= m;
            A.c1.y -= m;
            A.c2.z -= m;

            float p1 = (A.c0.y * A.c0.y + A.c0.z * A.c0.z + A.c1.z * A.c1.z);
        
            if (math.abs(p1) < EPSILON) {
                // 对角矩阵情况
                lambda = new float3(A.c0.x, A.c1.y, A.c2.z);
                V = float3x3.identity;
                return;
            }

            float q = math.determinant(A) * 0.5f;
            float p = (A.c0.x * A.c0.x + A.c1.y * A.c1.y + A.c2.z * A.c2.z +
                       2.0f * p1) / 6.0f;

            float phi = 1.0f / 3.0f;
            float r = math.sqrt(p * p * p);
            float3 eig;

            if (r > EPSILON)
            {
                float s = q / r;
                s = math.clamp(s, -1.0f, 1.0f);
                phi = math.acos(s) * phi;
                float twoSqrtP = 2.0f * math.sqrt(p);
                eig.x = twoSqrtP * math.cos(phi);
                eig.y = twoSqrtP * math.cos(phi - math.PI * 2.0f / 3.0f);
                eig.z = twoSqrtP * math.cos(phi - math.PI * 4.0f / 3.0f);
            }
            else
                eig = float3.zero;
        
            eig += m;
            lambda = eig;

            V = float3x3.identity;
            A = symCov;
            for (int i = 0; i < 3; ++i)
            {
                float lambda_i = lambda[i];

                // 构造 (A - λI) 矩阵
                float3 r0 = A.c0 - new float3(lambda_i, 0, 0);
                float3 r1 = A.c1 - new float3(0, lambda_i, 0);
                float3 r2 = A.c2 - new float3(0, 0, lambda_i);
                float3 v  = math.cross(r0, r1);
                if (math.length(v) < EPSILON) 
                    v = math.cross(r0, r2);
                if (math.length(v) < EPSILON) 
                    v = math.cross(r1, r2);
                if (math.length(v) < EPSILON)
                {
                    v = float3.zero;
                    v[i] = 1;
                }

                V[i] = math.normalize(v);
            }
        }
    
        public static void EVD_Jacobi(float3x3 A, out float3 lambda, out float3x3 V)
        {
            float3x3 D = A;
            V = float3x3.identity;
            for (int sweep = 0; sweep < 10; ++sweep) 
            {
                float maxOff = 0.0f;
                int p = 0, q = 1;

                // 1. 找最大非对角元
                for (int i = 0; i < 3; ++i)
                for (int k = i + 1; k < 3; ++k) 
                {
                    float aik = math.abs(D[i][k]);
                    if (aik < maxOff) 
                        continue;
                    maxOff = aik; 
                    p = i;
                    q = k;
                }
                if (maxOff < EPSILON) break;

                // 2. 计算 Jacobi 旋转角
                float App = D[p][p];
                float Aqq = D[q][q];
                float Apq = D[p][q];
                float tau = (Aqq - App) / (2.0f * Apq);
                float t = math.sign(tau) / (math.abs(tau) + math.sqrt(1.0f + tau * tau));
                float c = 1.0f / math.sqrt(1.0f + t * t);
                float s = t * c;

                // 3. 更新 D = J^T D J
                float3x3 J = float3x3.identity;
                J[p][p] = c;
                J[q][q] = c;
                J[p][q] = -s;
                J[q][p] = s;

                D = math.mul(math.transpose(J), math.mul(D, J));
                V = math.mul(V, J);
            }

            lambda = new float3(D[0][0], D[1][1], D[2][2]);
        }
    }
}
