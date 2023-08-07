#!/bin/bash

replacement_line=$'circle_img = self.image.copy()'
additional_line=$'cv2.circle(circle_img, (int(x), int(y)), 6, (0, 255, 0), 2)'

file_path=$'/usr/local/lib/python3.6/dist-packages/pop/Pilot.py'

matching_lines=$(grep -n 'cv2.circle(self.image, (int(x), int(y)), 6, (0, 255, 0), 2)' ${file_path})

if [ -n "$matching_lines" ]; then
    i=0
    
    while IFS= read -r line_info; do
        line_number=$(echo "$line_info" | cut -d ':' -f 1)
        line_number=$((line_number + i))
        line_number_add=$((line_number + 1))
        line_content=$(echo "$line_info" | cut -d ':' -f 2-)
        indent=$(echo "$line_content" | awk '{ match($0, /^[ \t]*/); print substr($0, RSTART, RLENGTH); }')
  
        replacement_content="${indent}${replacement_line}"
        additional_content="${indent}${additional_line}"
        sed -i "${line_number}s/.*/${replacement_content}\n/" ${file_path}
        sed -i "${line_number_add}s/.*/${additional_content}/" ${file_path}
        i=$((i + 1))
        
    done <<< "$matching_lines"
else
    echo "cv2.circle(self.image, (int(x), int(y)), 6, (0, 255, 0), 2) not found."
fi

replacement_line=$'self.imageWidget.value=bgr8_to_jpeg(circle_img)'
matching_lines=$(grep -n 'self.imageWidget.value=bgr8_to_jpeg(self.image)' ${file_path})

if [ -n "$matching_lines" ]; then    
    while IFS= read -r line_info; do
        line_number=$(echo "$line_info" | cut -d ':' -f 1)
        line_content=$(echo "$line_info" | cut -d ':' -f 2-)
        indent=$(echo "$line_content" | awk '{ match($0, /^[ \t]*/); print substr($0, RSTART, RLENGTH); }')
  
        replacement_content="${indent}${replacement_line}"
        sed -i "${line_number}s/.*/${replacement_content}\n/" ${file_path}
        
    done <<< "$matching_lines"
else
    echo "self.imageWidget.value=bgr8_to_jpeg(self.image) not found."
fi
