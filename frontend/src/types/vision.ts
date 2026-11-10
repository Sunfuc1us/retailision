export interface SKU {sku_id:string;brand:string;product:string;flavor:string;specification:string}
export interface Prediction {success:boolean;target_found:boolean;reason?:string;sku_id?:string;identity_score?:number;instances?:{instance_id:number;sku_id:string;identity_score:number;bbox:number[];mask_path:string}[];timing?:{total_ms:number};candidates?:unknown[]}
export interface Model {model_name:string;model_version:string;task?:string}
