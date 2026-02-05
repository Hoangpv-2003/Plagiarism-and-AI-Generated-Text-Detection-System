import React from 'react';

const HighlightedText = ({ sentenceAnalysis }) => {
  if (!sentenceAnalysis || sentenceAnalysis.length === 0) {
    return <p className="text-gray-500">Không có văn bản để hiển thị</p>;
  }

  const getHighlightColor = (confidence, isSuspicious) => {
    if (!isSuspicious) {
      return '';
    }
    
    if (confidence >= 0.7) {
      return 'bg-red-200 border-l-4 border-red-500 px-2 py-1';
    } else if (confidence >= 0.5) {
      return 'bg-orange-200 border-l-4 border-orange-500 px-2 py-1';
    } else if (confidence >= 0.3) {
      return 'bg-yellow-200 border-l-4 border-yellow-500 px-2 py-1';
    }
    return '';
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 0.7) {
      return 'High';
    } else if (confidence >= 0.5) {
      return 'Medium';
    } else if (confidence >= 0.3) {
      return 'Low';
    }
    return 'Very Low';
  };

  return (
    <div className="space-y-2">
      <div className="prose max-w-none">
        {sentenceAnalysis.map((item, index) => {
          const highlightClass = getHighlightColor(item.confidence, item.is_suspicious);
          
          return (
            <span key={index} className="inline">
              <span 
                className={`${highlightClass} ${item.is_suspicious ? 'rounded' : ''} transition-all hover:shadow-sm`}
                title={item.is_suspicious ? `Confidence: ${(item.confidence * 100).toFixed(1)}% | Source: ${item.source_title || 'Unknown'}` : ''}
              >
                {item.sentence}
              </span>
              {' '}
            </span>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <h4 className="text-sm font-semibold mb-3">Mức độ:</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-200 border-l-4 border-red-500 rounded"></div>
            <span>Cao (≥70%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-orange-200 border-l-4 border-orange-500 rounded"></div>
            <span>Trung bình (50-70%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-200 border-l-4 border-yellow-500 rounded"></div>
            <span>Thấp (30-50%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-white border-2 border-gray-300 rounded"></div>
            <span>Nguyên bản</span>
          </div>
        </div>
      </div>

      {/* Suspicious Sentences Details */}
      {sentenceAnalysis.filter(s => s.is_suspicious).length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-semibold mb-3">Chi tiết câu nghi ngờ:</h4>
          <div className="space-y-3">
            {sentenceAnalysis
              .filter(s => s.is_suspicious)
              .sort((a, b) => b.confidence - a.confidence)
              .map((item, index) => (
                <div 
                  key={index} 
                  className="p-3 bg-white border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between mb-2">
                    <span className={`inline-block px-2 py-1 rounded text-xs font-semibold ${
                      item.confidence >= 0.7 ? 'bg-red-100 text-red-800' :
                      item.confidence >= 0.5 ? 'bg-orange-100 text-orange-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {getConfidenceLabel(item.confidence)} Risk - {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-700 mb-2 italic">
                    "{item.sentence}"
                  </p>
                  
                  {item.source_title && (
                    <div className="text-xs text-gray-600">
                      <span className="font-semibold">Nguồn:</span> {item.source_title}
                    </div>
                  )}
                  
                  {item.source_url && (
                    <div className="text-xs mt-1">
                      <a 
                        href={item.source_url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 hover:underline"
                      >
                        Xem nguồn →
                      </a>
                    </div>
                  )}
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default HighlightedText;
