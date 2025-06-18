import requests
import json

# API 服务器的地址
API_URL = "http://localhost:8000/v1/chat/completions"

# 准备请求头
headers = {
    "Content-Type": "application/json"
}

# --- 控制开关 ---
# 你可以在这里切换 True 或 False 来测试两种不同的模式
SHOULD_STREAM = True
code = '''
// GaussianClustering.cpp : implementation of the TGaussianClustering class
//
// Class for algorithms that perform simple clustering with Gaussian statistics.
//
/////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "GaussianClustering.h"
#include "EstimateUtil.h"

#define REG_COEF 0.001

//Threshold for detecting a zero value 10^(-12)
#define ZERO_THRESH 1e-12

//
// Constructor
// 
TGaussianClustering::TGaussianClustering( int nClusterCnt)
	                  : TSimpleClustering(nClusterCnt)
{
	m_ppMu = 0;
	m_ppCovInv = 0;
	m_dbPri = 0;
	m_dbDeterm = 0;
	m_nFeatureCnt = 0;

	//srand((unsigned) time(NULL));
}

//
// Destructor
// 
TGaussianClustering::~TGaussianClustering()
{
    Clear();

}

/// Get segmentation statistics object that lists all the class statistics.
/**
 * The statistics will not be updated before being returned
 */
TSegmentationStatistics* TGaussianClustering::GetSegStats()
{
	if (m_nClusterCnt <= 0 || m_nFeatureCnt <= 0)
		return 0;

	TGaussianSegmentationStatistics tscc(m_nClusterCnt);

	TMatrix* temp = 0;

	for (int k = 0; k < m_nClusterCnt; k++)
	{
		if (m_ppMu[k] && m_ppCovInv[k && m_nClassPixelCnt])
		{
			tscc.m_vMu[k] = *m_ppMu[k];
			temp = m_ppCovInv[k]->Inv();
			tscc.m_vCov[k] = *temp;
			delete temp;
			tscc.m_vPixelCounts[k] = m_nClassPixelCnt[k];
		}
		else
		{
			return 0;
		}
	}
	
	//return new TGaussianSegmentationStatistics(tscc);
	return tscc.Clone();
}


bool TGaussianClustering::Alloc()
{
	if ( m_nClusterCnt <= 0 || m_nFeatureCnt <= 0 )
		return false;
	//if ( m_nClusterCnt == nClusterCnt && m_nFeatureCnt == nFeatureCnt )
	//	return true;

	//Clear();

	//m_nClusterCnt = nClusterCnt;
	//m_nFeatureCnt = nFeatureCnt;

	m_ppMu = AllocVectors(m_nClusterCnt, m_nFeatureCnt);
	m_ppCovInv = AllocMatrice(m_nClusterCnt, m_nFeatureCnt);
	m_dbDeterm = new double[m_nClusterCnt];
	m_dbPri = new double[m_nClusterCnt];

	return true;
}

// 
// Clear memory
// 
void TGaussianClustering::Clear()
{

	if (m_ppMu)
	{
		for ( int i = 0; i < m_nClusterCnt; i++ )
			delete m_ppMu[i];
		delete[] m_ppMu;
		m_ppMu = 0;
	}
	if (m_ppCovInv)
	{
		for ( int i = 0; i < m_nClusterCnt; i++ )
			delete m_ppCovInv[i];
		delete[] m_ppCovInv;
		m_ppCovInv = 0;
	}
	if (m_dbDeterm)
	{
		delete[] m_dbDeterm;
		m_dbDeterm = 0;
	}
	if (m_dbPri)
	{
		delete[] m_dbPri;
		m_dbPri = 0;
	}

	m_nFeatureCnt = 0;

}

//
// Allocate vectors (for means) for a given number of clusters
// 
TVector** TGaussianClustering::AllocVectors(int nClusterCnt, int nFeatureCnt)
{
	TVector** ppRet = 0;
	if ( nClusterCnt > 0 && nFeatureCnt > 0 )
	{
		ppRet = new TVector*[nClusterCnt];
		for ( int i = 0; i < nClusterCnt; i++ )
			ppRet[i] = new TVector(nFeatureCnt, false);  // row vector
	}
	return ppRet;
}

//
// Allocate matrice (for covariance) for a given number of clusters
// 
TMatrix** TGaussianClustering::AllocMatrice(int nClusterCnt, int nFeatureCnt)
{
	TMatrix** ppRet = 0;
	if ( nClusterCnt > 0 && nFeatureCnt > 0 )
	{
		ppRet = new TMatrix*[nClusterCnt];
		for ( int i = 0; i < nClusterCnt; i++ )
			ppRet[i] = new TMatrix(nFeatureCnt, nFeatureCnt);
	}
	return ppRet;
}

// 
// Set all elements in the vectors to be zero
// 
void TGaussianClustering::SetZeroVector(TVector** ppVectors, int nClusterCnt)
{
	for ( int i = 0; i < nClusterCnt; i++ )
		ppVectors[i]->SetZero();
}

// 
// Set all elements in the matrice to be zero
// 
void TGaussianClustering::SetZeroMatrix(TMatrix** ppMatrice, int nClusterCnt)
{
	for ( int i = 0; i < nClusterCnt; i++ )
		ppMatrice[i]->SetZero();
}

//
// Arrange the clusters in the ascending order of the first mean feature values, so that the resulting map looks more pleasant
// 
void TGaussianClustering::SortCluster()
{
	TVector* pVecSwap;
	TMatrix* pMatSwap;
	double dbPriSwap, dbDetermSwap;
	for ( int i = 0; i < m_nClusterCnt; i++ )
		for ( int j = i + 1; j < m_nClusterCnt; j++ )
		{
			if ( (*m_ppMu[i])(0) > (*m_ppMu[j])(0) )
			{
				pVecSwap = m_ppMu[i]; m_ppMu[i] = m_ppMu[j]; m_ppMu[j] = pVecSwap;
				pMatSwap = m_ppCovInv[i]; m_ppCovInv[i] = m_ppCovInv[j]; m_ppCovInv[j] = pMatSwap;
				dbPriSwap = m_dbPri[i]; m_dbPri[i] = m_dbPri[j]; m_dbPri[j] = dbPriSwap;
				dbDetermSwap = m_dbDeterm[i]; m_dbDeterm[i] = m_dbDeterm[j]; m_dbDeterm[j] = dbDetermSwap;
			}
		}
}

// 
// Display the cluster parameters
// 
void TGaussianClustering::ShowParam()
{
	int nFeatureCnt = m_nFeatureCnt;
	int i, j, k;
	for ( k = 0; k < m_nClusterCnt; k++ )
	{
		cout << "Mu: ";
		for ( i = 0; i < nFeatureCnt; i++ )
			cout << (*m_ppMu[k])(i) << ", ";

		cout << "Cov: ";
		for ( i = 0; i < nFeatureCnt; i++ )
			for ( j = 0; j < nFeatureCnt; j++ )
				cout << (*m_ppCovInv[k])(i, j) << ", ";

		cout << "Pri: " << m_dbPri[k] << endl;
	}
}
'''
# 准备请求体
payload = {
    "model": "qwen2.5-0.5b-instruct",
    "messages": [
        {"role": "system", "content": "你是一个专业的软件工程师。"},
        {"role": "user", "content": f"分析一下这段代码 {code}"}
    ],
    "stream": SHOULD_STREAM,
    "temperature": 0.7
}

print(f"正在向服务器发送 {'流式' if SHOULD_STREAM else '常规'} 请求...")

try:
    # 发送POST请求，注意 stream 参数会根据 SHOULD_STREAM 的值变化
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload), stream=SHOULD_STREAM)
    response.raise_for_status() # 检查请求是否成功

    # ======================================================
    #  根据 stream 的值，选择不同的处理方式
    # ======================================================

    if SHOULD_STREAM:
        # --- 流式处理逻辑 ---
        print("\n服务器实时返回的回答：")
        # 迭代处理服务器返回的每一行数据
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_str = decoded_line[6:]
                    if json_str == '[DONE]':
                        break
                    try:
                        data = json.loads(json_str)
                        delta_content = data['choices'][0]['delta'].get('content', '')
                        print(delta_content, end='', flush=True)
                    except json.JSONDecodeError:
                        pass
        # 打印一个换行，让终端显示更美观
        print()

    else:
        # --- 常规处理逻辑 ---
        # 一次性解析服务器返回的完整JSON数据
        response_data = response.json()
        print("\n服务器返回的完整回答：")
        # 打印出模型完整的回答
        print(response_data['choices'][0]['message']['content'])


except requests.exceptions.RequestException as e:
    print(f"\n请求失败: {e}")
